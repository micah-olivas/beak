"""Per-sequence detail modal pushed from the alignment view.

Surfaces project-scoped context for one row of the alignment: identity
to the target, taxonomy lineage + growth temperature, structure
availability, and any populated traits from `traits.parquet`. Pure
local — no SSH, no remote calls. Reads the active homolog set's
parquets on open (~50 ms total even on a 25k-row taxonomy table).
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ...project import BeakProject


# Lineage rank order for the breadcrumb display, broadest → narrowest.
# Skips `superkingdom` since `domain` carries the same value via the
# legacy alias in `tui/taxonomy.py`.
_LINEAGE_RANKS: List[str] = [
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

# Cap on the per-row label width so a long UniRef accession doesn't
# push the residue block far to the right; longer ids truncate.
_PAIRWISE_LABEL_W_MAX = 16

# Colours for the residue rows — separate from the main alignment-
# view biochem palette on purpose. Here the comparison axis (match vs
# similar vs mismatch vs gap) is the information being encoded; biochem
# identity would just add visual noise.
_PAIRWISE_COLOURS = {
    "match":     "#65CBF3",  # cyan — exact identity
    "similar":   "#7DD87D",  # green — non-identical but BLOSUM62 score > 0
    "mismatch":  "#FFA62B",  # amber — zero or negative BLOSUM62 score
}

# Sentinel for the taxonomy-row cache — `None` is a legitimate "row not
# found" result, so we need a separate "not yet loaded" marker.
_NOT_LOADED = object()


def _blosum62_score(a: str, b: str) -> int:
    """Score of (a, b) under BLOSUM62, 0 for any character pair the
    matrix doesn't cover (selenocysteine `U`, stops, etc.).

    Loaded lazily on first call so the modal's import cost stays at
    zero until someone actually opens it. The matrix object itself is
    a few KB and cached on the function for subsequent calls.
    """
    matrix = getattr(_blosum62_score, "_matrix", None)
    if matrix is None:
        try:
            from Bio.Align import substitution_matrices
            matrix = substitution_matrices.load("BLOSUM62")
        except Exception:
            # Sentinel: remember the load failure so we don't retry on
            # every comparison. Plain `False` since the matrix object
            # is array-like and truthy-checks raise on it.
            matrix = False
        _blosum62_score._matrix = matrix
    if matrix is False:
        return 0
    try:
        return int(matrix[a.upper(), b.upper()])
    except (KeyError, IndexError, ValueError):
        return 0


class SequenceDetailModal(ModalScreen[None]):
    """One-shot inspect of a single sequence in the alignment."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    DEFAULT_CSS = """
    /* Width is set in app.py's global modal CSS — keeping it there
       gives equal specificity with the global `width: 64` rule, which
       it overrides via source order. */
    /* Two-axis scroller so the pairwise block can extend off the
       right of the visible window without wrapping into a clutter
       of stacked chunks. */
    SequenceDetailModal #seq-detail-scroll {
        height: 1fr;
        max-height: 28;
        overflow-x: auto;
        overflow-y: auto;
    }
    SequenceDetailModal .section {
        margin-top: 1;
    }
    /* Pairwise pane needs `width: auto` so its content (which can be
       hundreds of cells wide) drives the natural width — the parent
       scroller then has something to pan over. */
    SequenceDetailModal #seq-detail-pairwise {
        width: auto;
    }
    """

    def __init__(
        self,
        project: BeakProject,
        seq_id: str,
        aligned_seq: str,
        target_id: str,
        target_aligned: str,
        is_target: bool = False,
        tax_rows_by_id: Optional[dict] = None,
        traits_rows_by_id: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._seq_id = seq_id
        self._aligned_seq = aligned_seq
        self._target_id = target_id
        self._target_aligned = target_aligned
        self._is_target = is_target
        # Pre-warmed lookup tables from the alignment view. `None` =
        # warm not yet finished — fall back to disk read in that case.
        self._tax_rows_by_id = tax_rows_by_id
        self._traits_rows_by_id = traits_rows_by_id
        # Cache for the taxonomy row — taxonomy + structure sections
        # both want it; reading the parquet twice cost ~230 ms on a
        # 25k-row table. `_NOT_LOADED` distinguishes "not yet looked
        # up" from "looked up and missing".
        self._tax_row_cache: object = _NOT_LOADED

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(self._title_text(), id="modal-title")
            with ScrollableContainer(id="seq-detail-scroll"):
                # Identity is O(L) on aligned strings — render
                # immediately so the modal has visible content the
                # instant it opens.
                yield Static(
                    self._render_identity_section(),
                    id="seq-detail-identity",
                )
                # Heavy sections (pairwise BLOSUM, taxonomy/traits
                # parquet reads) are filled in by a background worker
                # — see `_populate_async`. Placeholders give the user
                # a visible "loading" state instead of a blank gap.
                if not self._is_target:
                    yield Static(
                        "[dim]computing pairwise alignment…[/dim]",
                        id="seq-detail-pairwise",
                    )
                yield Static(
                    "[dim]loading taxonomy…[/dim]",
                    id="seq-detail-taxonomy",
                    classes="section",
                )
                # Traits and structure sections start hidden — the
                # async loader will reveal them only if there is
                # content to show, avoiding empty gaps in the modal
                # layout while the worker runs.
                traits_ph = Static(
                    "", id="seq-detail-traits", classes="section",
                )
                traits_ph.display = False
                yield traits_ph
                struct_ph = Static(
                    "", id="seq-detail-structure", classes="section",
                )
                struct_ph.display = False
                yield struct_ph
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close-btn", variant="primary")

    def on_mount(self) -> None:
        # Kick off the slow sections in a thread worker — keeps the
        # modal frame paint snappy (~50 ms on the main thread instead
        # of ~750 ms with everything inline).
        self._populate_async()

    @work(thread=True, exclusive=True, group="seq-detail-load")
    def _populate_async(self) -> None:
        # Pairwise: cheap O(L) but BLOSUM62 first-load is ~500 ms.
        if not self._is_target:
            pw = self._pairwise_text()
            self.app.call_from_thread(self._set_section, "seq-detail-pairwise", pw)
        else:
            self.app.call_from_thread(self._hide_section, "seq-detail-pairwise")

        # Taxonomy + structure share the same parquet — load once.
        tax = self._render_taxonomy_section()
        self.app.call_from_thread(self._set_section, "seq-detail-taxonomy", tax)
        struct = self._render_structure_section()
        self.app.call_from_thread(self._set_section, "seq-detail-structure", struct)

        # Traits last — traits.parquet is the widest table (often 2k+
        # columns) so this is the slowest read.
        traits = self._render_traits_section()
        self.app.call_from_thread(self._set_section, "seq-detail-traits", traits)

    def _set_section(self, widget_id: str, content) -> None:
        try:
            w = self.query_one(f"#{widget_id}", Static)
        except Exception:
            return
        if not content:
            w.display = False
            return
        w.update(content)
        w.display = True

    def _hide_section(self, widget_id: str) -> None:
        try:
            self.query_one(f"#{widget_id}", Static).display = False
        except Exception:
            pass

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.action_close()

    # ---- rendering ----

    def _title_text(self) -> str:
        tag = "[bold #2E86AB]target[/bold #2E86AB]" if self._is_target else "homolog"
        return f"[bold]{self._seq_id}[/bold]  [dim]·[/dim]  {tag}"


    def _render_identity_section(self) -> str:
        # All identity stats are O(L) on the aligned strings, no I/O.
        n_overlap, n_match, n_target_aa, n_seq_aa = _identity_counts(
            self._target_aligned, self._aligned_seq
        )
        rows = ["[bold]Identity[/bold]"]
        if self._is_target:
            rows.append(
                f"  [dim]This row is the target sequence "
                f"({n_seq_aa} aa).[/dim]"
            )
            return "\n".join(rows)
        if n_overlap == 0:
            rows.append("  [dim]No ungapped overlap with target.[/dim]")
        else:
            pct = n_match / n_overlap * 100.0
            rows.append(
                f"  [bold]{pct:5.1f}%[/bold]  "
                f"[dim]identity over {n_overlap:,} overlapping positions "
                f"({n_match:,} matches)[/dim]"
            )
        # Gap fraction relative to the alignment width — high gap-frac
        # rows are usually fragments or distantly-related hits.
        aln_w = max(1, len(self._aligned_seq))
        n_gaps = sum(1 for c in self._aligned_seq if c in "-.")
        gap_pct = n_gaps / aln_w * 100.0
        rows.append(
            f"  [dim]ungapped length:[/dim] [bold]{n_seq_aa:,}[/bold] aa  "
            f"[dim]·  alignment gaps:[/dim] [bold]{gap_pct:.0f}%[/bold]"
        )
        return "\n".join(rows)

    def _pairwise_text(self) -> Optional[Text]:
        """BLAST-style three-line pairwise, all on a single horizontal
        run for each row (no chunking).

        The Rich Text returned is constructed with ``no_wrap=True`` so
        Textual's renderer doesn't soft-wrap a 600-aa alignment into
        cluttered stacked blocks — the parent ``ScrollableContainer``
        provides horizontal panning instead.

        Layout (single line per axis):
            <target_label>   1  RESIDUES…………………  587
                                |  | || ||  | …
            <hit_label>      1  RESIDUES…………………  514

        Match marks render as plain `|` (default text colour) per the
        UX call: amber/cyan residues already encode the comparison
        axis on the residue rows, so a high-contrast, low-saturation
        match line keeps the visual hierarchy clean.
        """
        t_aln = self._target_aligned
        h_aln = self._aligned_seq
        n = min(len(t_aln), len(h_aln))
        if n == 0:
            return None

        # Pre-compute per-column comparison class. `similar` is the
        # BLOSUM62 conservative-substitution tier: BLAST renders these
        # as `+` in its "Positives" line. Here we use `:` to keep the
        # match line monospaced and visually distinct from `|` (exact)
        # without introducing a third row.
        classes: List[str] = []
        # ^ "match" | "similar" | "mismatch" | "tgap" | "hgap" | "both_gap"
        for i in range(n):
            t = t_aln[i]
            h = h_aln[i]
            t_gap = t in "-."
            h_gap = h in "-."
            if t_gap and h_gap:
                classes.append("both_gap")
            elif t_gap:
                classes.append("tgap")
            elif h_gap:
                classes.append("hgap")
            elif t.upper() == h.upper():
                classes.append("match")
            elif _blosum62_score(t, h) > 0:
                classes.append("similar")
            else:
                classes.append("mismatch")

        n_ungapped = sum(
            1 for c in classes if c in ("match", "similar", "mismatch")
        )
        n_match = sum(1 for c in classes if c == "match")
        n_similar = sum(1 for c in classes if c == "similar")
        n_positives = n_match + n_similar
        pct = (n_match / n_ungapped * 100.0) if n_ungapped else 0.0
        pct_pos = (n_positives / n_ungapped * 100.0) if n_ungapped else 0.0

        # Label width — fits the longest of {target_id, hit_id} but
        # capped so a long UniRef accession doesn't push the residue
        # block out to col 40.
        label_w = max(len(self._target_id), len(self._seq_id), 6)
        label_w = min(label_w, _PAIRWISE_LABEL_W_MAX)
        t_label = self._target_id[:label_w].ljust(label_w)
        h_label = self._seq_id[:label_w].ljust(label_w)

        # Walk the columns once, accumulating glyphs + ungapped
        # position counters per side.
        t_first = t_last = None
        h_first = h_last = None
        t_residues: List[tuple] = []  # (char, style)
        h_residues: List[tuple] = []
        match_chars: List[str] = []
        t_pos = 1
        h_pos = 1
        for i in range(n):
            cls = classes[i]
            t = t_aln[i]
            h = h_aln[i]

            if cls in ("tgap", "both_gap"):
                t_residues.append(("-", "dim"))
            else:
                t_residues.append((t, _PAIRWISE_COLOURS.get(cls, "")))
                if t_first is None:
                    t_first = t_pos
                t_last = t_pos
                t_pos += 1

            if cls in ("hgap", "both_gap"):
                h_residues.append(("-", "dim"))
            else:
                h_residues.append((h, _PAIRWISE_COLOURS.get(cls, "")))
                if h_first is None:
                    h_first = h_pos
                h_last = h_pos
                h_pos += 1

            if cls == "match":
                match_chars.append("|")
            elif cls == "similar":
                match_chars.append(":")
            else:
                match_chars.append(" ")

        text = Text(no_wrap=True)
        text.append("Pairwise to target", style="bold")
        # Two summary stats now: "identity" (exact matches) and
        # "positives" (identity + BLOSUM62-conservative substitutions),
        # mirroring BLAST's two-line summary. Legend names every
        # match-line glyph so a reader doesn't have to remember which
        # is which.
        text.append(
            f"  ({pct:.1f}% identity · "
            f"{pct_pos:.1f}% positives over {n_ungapped:,} positions · "
            f"| exact · : BLOSUM62+ · space mismatch · - gap)\n",
            style="dim",
        )

        # Header offset (label + space + 5-digit counter + spaces) so
        # the match line and counters line up under the residues. 14 =
        # leading 2 spaces + label_w + space + 5-digit + 2 trailing.
        prefix_w = 2 + label_w + 1 + 5 + 2
        # Target row.
        text.append("  ")
        text.append(t_label, style="#2E86AB bold")
        text.append(f" {(t_first or 0):>5}  ", style="dim")
        _append_runs(text, t_residues)
        text.append(f"  {(t_last or 0):>5}\n", style="dim")
        # Match row — plain (default colour) `|` for matches per UX
        # request. Indent by `prefix_w` so columns line up with the
        # residue rows above and below.
        text.append(" " * prefix_w)
        text.append("".join(match_chars))
        text.append("\n")
        # Hit row.
        text.append("  ")
        text.append(h_label, style="dim")
        text.append(f" {(h_first or 0):>5}  ", style="dim")
        _append_runs(text, h_residues)
        text.append(f"  {(h_last or 0):>5}", style="dim")
        return text

    def _render_taxonomy_section(self) -> str:
        row = self._lookup_taxonomy_row()
        if row is None:
            return ""
        rows = ["[bold]Taxonomy[/bold]"]
        # Lineage breadcrumbs — render the populated ranks separated by
        # a thin chevron. Skip the column entirely when no rank is set
        # so a sparsely-resolved row doesn't render as a wall of "—".
        crumbs: List[str] = []
        for rank in _LINEAGE_RANKS:
            v = row.get(rank)
            if pd.notna(v) and str(v).strip():
                crumbs.append(str(v))
        if crumbs:
            rows.append("  " + "  ›  ".join(crumbs))
        else:
            rows.append("  [dim](lineage not resolved)[/dim]")

        organism = row.get("organism")
        if pd.notna(organism) and str(organism).strip():
            rows.append(f"  [dim]organism:[/dim] {organism}")

        # Growth temperature with provenance — `temp_source` says
        # whether this number was a direct species hit in the Enqvist
        # dataset or a genus-level fallback (median across the genus).
        gt = row.get("growth_temp")
        if pd.notna(gt):
            ts = row.get("temp_source")
            ts_label = (
                f"  [dim](source: {ts})[/dim]"
                if pd.notna(ts) and ts else ""
            )
            rows.append(
                f"  [dim]growth temp:[/dim] [bold]{float(gt):.0f} °C[/bold]"
                f"{ts_label}"
            )

        # Where the rank/organism came from — UniProt-authoritative
        # rows weight more than mmseqs LCA inferences.
        tax_src = row.get("taxonomy_source")
        if pd.notna(tax_src) and str(tax_src).strip():
            rows.append(f"  [dim]taxonomy source:[/dim] {tax_src}")
        return "\n".join(rows)

    def _render_traits_section(self) -> str:
        # Prefer the in-memory cache pre-loaded by the alignment view
        # (~150 ms saved per modal open on the Em_PafA-scale traits
        # table). Falls back to a direct parquet read only when the
        # warm worker hasn't finished yet.
        row: Optional[dict] = None
        if self._traits_rows_by_id is not None:
            row = self._traits_rows_by_id.get(str(self._seq_id))
            if row is None:
                from ..taxonomy import _accession_from_seq_id
                acc = _accession_from_seq_id(self._seq_id)
                if acc:
                    for r in self._traits_rows_by_id.values():
                        if str(r.get("uniprot_id", "")) == acc:
                            row = r
                            break
        else:
            traits_path = self._active_homologs_dir() / "traits.parquet"
            if not traits_path.exists():
                return ""
            try:
                df = pd.read_parquet(traits_path)
            except Exception:
                return ""
            row = _row_for_seq_id(df, self._seq_id)
        if row is None:
            return ""

        # Surface only trait_* columns that have a non-null, non-empty
        # value for this row. NA / blank traits are noise on a single-
        # sequence view; the per-set distribution lives in the taxonomy
        # view's "Trait composition" pane.
        kvs: List[str] = []
        for col, val in row.items():
            if not str(col).startswith("trait_"):
                continue
            if col == "trait_match_level":
                continue  # bookkeeping column, not a trait
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if not sval or sval.lower() in ("nan", "none"):
                continue
            label = str(col).removeprefix("trait_").replace("_", " ")
            # Pretty-print boolean-ish values with check / cross marks.
            low = sval.lower()
            if low == "true":
                pretty = "[green]✓[/green]"
            elif low == "false":
                pretty = "[red]✗[/red]"
            else:
                pretty = sval
            kvs.append(f"  [dim]{label}:[/dim] {pretty}")
        if not kvs:
            return ""

        match_level = row.get("trait_match_level")
        header = "[bold]Traits[/bold]"
        if pd.notna(match_level) and str(match_level).strip():
            header += f"  [dim](match level: {match_level})[/dim]"
        return header + "\n" + "\n".join(kvs)

    def _render_structure_section(self) -> str:
        # Only meaningful for UniProt-resolvable hits. Show "available"
        # when the AlphaFold CIF is already cached; otherwise skip the
        # section entirely (avoids a misleading "no structure" line for
        # a hit that simply hasn't been queried).
        row = self._lookup_taxonomy_row()
        uniprot_id = None
        if row is not None:
            v = row.get("uniprot_id")
            if pd.notna(v) and str(v).strip():
                uniprot_id = str(v)
        if not uniprot_id:
            return ""
        # Prefer experimental PDB over AlphaFold — labels the cache hit
        # with the source so the user knows which model they'd open.
        from ..structure import _cached_structure
        cached = _cached_structure(
            uniprot_id, self._project.path / "structures",
        )
        if cached is not None:
            cif_path, source_label = cached
            sz = cif_path.stat().st_size
            return (
                f"[bold]Structure[/bold]\n"
                f"  [green]✓[/green] {source_label} cached  "
                f"[dim]({sz / 1024:.0f} KB · {cif_path.name})[/dim]"
            )
        return (
            f"[bold]Structure[/bold]\n"
            f"  [dim]not fetched · open the project to pull a "
            f"PDB or AlphaFold model[/dim]"
        )

    # ---- helpers ----

    def _active_homologs_dir(self) -> Path:
        return self._project.active_homologs_dir()

    def _lookup_taxonomy_row(self):
        """Return a dict-like row from `taxonomy.parquet` for the seq,
        or None if the table is missing or doesn't contain the id.

        Prefers the warm cache pre-loaded by the alignment view. Falls
        back to a direct parquet read only when the warm worker hasn't
        finished yet (rare on a warm interpreter).
        """
        if self._tax_row_cache is not _NOT_LOADED:
            return self._tax_row_cache
        if self._tax_rows_by_id is not None:
            row = self._tax_rows_by_id.get(str(self._seq_id))
            if row is None:
                # Try uniprot_id fallback — same logic as
                # `_row_for_seq_id` but on the in-memory dict.
                from ..taxonomy import _accession_from_seq_id
                acc = _accession_from_seq_id(self._seq_id)
                if acc:
                    for r in self._tax_rows_by_id.values():
                        if str(r.get("uniprot_id", "")) == acc:
                            row = r
                            break
            self._tax_row_cache = row
            return row
        tax_path = self._active_homologs_dir() / "taxonomy.parquet"
        if not tax_path.exists():
            self._tax_row_cache = None
            return None
        try:
            df = pd.read_parquet(tax_path)
        except Exception:
            self._tax_row_cache = None
            return None
        row = _row_for_seq_id(df, self._seq_id)
        self._tax_row_cache = row
        return row


def _append_runs(text: Text, residues: List[tuple]) -> None:
    """Append (char, style) pairs to ``text`` collapsing consecutive
    same-style entries into a single ``Text.append`` call.

    Naive char-by-char appending of a 3,800-column pairwise spent ~40
    ms in Rich's segment list mutation; collapsing to runs cuts that
    by ~10x at the cost of two extra ints in the inner loop.
    """
    if not residues:
        return
    run_chars = [residues[0][0]]
    run_style = residues[0][1]
    for ch, style in residues[1:]:
        if style == run_style:
            run_chars.append(ch)
        else:
            text.append("".join(run_chars), style=run_style)
            run_chars = [ch]
            run_style = style
    text.append("".join(run_chars), style=run_style)


def _identity_counts(
    target_aligned: str, hit_aligned: str
) -> tuple:
    """Return ``(n_overlap, n_match, n_target_aa, n_hit_aa)``.

    Walks the two aligned strings in lockstep:
      - `n_overlap`: positions where both have a non-gap character
      - `n_match`:   non-gap positions where the residues are equal
                     (case-insensitive)
      - `n_target_aa` / `n_hit_aa`: total non-gap chars in each row,
                     used elsewhere to render "ungapped length".
    """
    n_overlap = n_match = n_target_aa = n_hit_aa = 0
    n = min(len(target_aligned), len(hit_aligned))
    for i in range(n):
        t = target_aligned[i]
        h = hit_aligned[i]
        t_is_gap = t in "-."
        h_is_gap = h in "-."
        if not t_is_gap:
            n_target_aa += 1
        if not h_is_gap:
            n_hit_aa += 1
        if not t_is_gap and not h_is_gap:
            n_overlap += 1
            if t.upper() == h.upper():
                n_match += 1
    # Account for any tail past the shorter string in the hit count
    # (rarely fires on real MSAs but keeps the math honest).
    if len(hit_aligned) > n:
        for ch in hit_aligned[n:]:
            if ch not in "-.":
                n_hit_aa += 1
    return n_overlap, n_match, n_target_aa, n_hit_aa


def _row_for_seq_id(df: pd.DataFrame, seq_id: str) -> Optional[dict]:
    """Find the parquet row matching `seq_id`, with a couple of
    accession-extraction fallbacks for FASTA id formats the parquet
    might not have stored verbatim (e.g., `UniRef90_P00533` vs `P00533`)."""
    if "sequence_id" not in df.columns:
        return None
    hit = df[df["sequence_id"].astype(str) == str(seq_id)]
    if hit.empty:
        # Fallback: extract the bare UniProt accession and match on
        # `uniprot_id` — handles cases where the alignment uses a
        # decorated id like `sp|P00533|EGFR_HUMAN` while the parquet
        # carries just `P00533`.
        from ..taxonomy import _accession_from_seq_id
        acc = _accession_from_seq_id(seq_id)
        if acc and "uniprot_id" in df.columns:
            hit = df[df["uniprot_id"].astype(str) == acc]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()
