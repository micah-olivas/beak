"""Modal showing detailed info for a project layer."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Select, Static

from ...project import BeakProject


_JOBS_DB = Path.home() / ".beak" / "jobs.json"


def _matchmaker_superpose(ref_coords, ref_seq: str,
                          mob_coords, mob_seq: str):
    """Sequence-aligned Kabsch superposition à la ChimeraX matchmaker.

    Pairwise-aligns ``mob_seq`` to ``ref_seq`` (BLOSUM62 global, default
    Biopython gap penalties), pulls the matched-residue CA pairs, and
    runs Kabsch to find the rigid (rotation + translation) that
    minimizes RMSD on those pairs. Returns
    ``(aligned_coords, rmsd, R, t)`` where ``aligned = mob @ R.T + t``,
    or ``(None, None, None, None)`` if alignment quality is too poor
    (<4 matched residues — Kabsch needs at least 3 non-collinear
    points and we want one extra for stability).

    One round of outlier pruning is run after the initial superposition:
    pairs whose post-fit pair distance exceeds 2× the per-pair RMSD are
    dropped and the fit is recomputed. This is a stripped-down version
    of Chimera's iterative pruning — enough to handle one or two
    misaligned loops without going full converge-on-core.
    """
    import numpy as np
    if (
        ref_coords is None or mob_coords is None
        or len(ref_seq) != len(ref_coords)
        or len(mob_seq) != len(mob_coords)
    ):
        return None, None, None, None
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError:
        return None, None, None, None

    aligner = PairwiseAligner()
    aligner.mode = "global"
    try:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        # Fall back to a simple match/mismatch scheme; less accurate
        # for distant homologs but doesn't hard-fail.
        aligner.match_score = 2
        aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    try:
        alignments = aligner.align(ref_seq, mob_seq)
        # Biopython's `aligned` attr is the most direct way to extract
        # matched-residue index pairs without parsing the gapped strings.
        # Format: ((ref_block_start, ref_block_end), (mob_block_start, ...))
        # for each contiguous gap-free block.
        first = alignments[0]
        ref_idxs: list = []
        mob_idxs: list = []
        for (rs, re), (ms, me) in zip(*first.aligned):
            block_len = min(re - rs, me - ms)
            for k in range(block_len):
                ref_idxs.append(rs + k)
                mob_idxs.append(ms + k)
    except Exception:
        return None, None, None, None

    if len(ref_idxs) < 4:
        return None, None, None, None

    ref_pts = ref_coords[ref_idxs]
    mob_pts = mob_coords[mob_idxs]

    def _kabsch(P, Q):
        """Return (R, t) such that P @ R.T + t ≈ Q (mob → ref)."""
        cP = P.mean(axis=0)
        cQ = Q.mean(axis=0)
        Pc = P - cP
        Qc = Q - cQ
        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.eye(3)
        D[2, 2] = d  # mirror correction → proper rotation
        R = Vt.T @ D @ U.T
        t = cQ - cP @ R.T
        return R, t

    R, t = _kabsch(mob_pts, ref_pts)
    aligned = mob_coords @ R.T + t

    # One round of outlier pruning. Pair distances > 2× the RMSD get
    # dropped; if at least 4 pairs survive, refit Kabsch on those.
    fitted = aligned[mob_idxs]
    diffs = fitted - ref_pts
    dists = np.linalg.norm(diffs, axis=1)
    rmsd = float(np.sqrt((dists ** 2).mean()))
    keep = dists < max(2.0 * rmsd, 1.0)
    if keep.sum() >= 4 and keep.sum() < len(dists):
        R, t = _kabsch(mob_pts[keep], ref_pts[keep])
        aligned = mob_coords @ R.T + t
        # Recompute final RMSD over the pruned set for the title.
        fitted = aligned[np.array(mob_idxs)[keep]]
        dists = np.linalg.norm(fitted - ref_pts[keep], axis=1)
        rmsd = float(np.sqrt((dists ** 2).mean()))

    # Return (R, t) alongside the aligned coords so callers (the
    # ligand-overlay path in the gallery) can apply the *same* rigid
    # transform to attached non-polymer atoms. The earlier code
    # recovered an effective `R_kab` from the aligned coords via
    # `np.linalg.lstsq` after the fact — that's mathematically wrong
    # because the input wasn't centered on the same origin as the
    # output, so the solution absorbed translation into a non-
    # orthogonal matrix and ligands ended up displaced from their
    # parent protein. Threading (R, t) out fixes the math.
    return aligned, rmsd, R, t


def _pca_canonical_orient(coords):
    """Rotate a CA point cloud into a canonical PCA frame.

    Without this, switching between two CIFs in the gallery showed
    each in its own arbitrary cif coordinate system — visually
    incomparable. PCA-aligning every structure puts:
      * longest axis horizontal (x)
      * second-longest vertical (y)
      * shortest into the page (z, which the depth-mist uses)
    and a sign convention that anchors the N-terminus on the left so
    consecutive structures read N→C the same way.

    Returns a centered + rotated (N, 3) array; the original is
    untouched. SVD-based, so it works for any chain length and is
    stable to noise. The full transform is a proper rotation
    (det = +1), so chirality is preserved across structures.
    """
    import numpy as np
    if coords is None or len(coords) < 2:
        return coords
    centered = coords - coords.mean(axis=0)
    # SVD on the point cloud: rows of Vt are the principal axes.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # Force a proper rotation. det(Vt) is ±1; if it's -1 we'd be
    # reflecting (mirror image), which would flip a left-handed helix
    # to a right-handed one between two otherwise-identical CIFs.
    if np.linalg.det(Vt) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
    aligned = centered @ Vt.T
    # Sign convention: N-terminus on -x (left), C-terminus on +x.
    # Compare the centroids of the first and second halves along the
    # first principal axis; flip x (and y, to keep it a proper
    # rotation) if the N-terminus came out on the right.
    n_half = max(1, len(aligned) // 2)
    if aligned[:n_half, 0].mean() > aligned[n_half:, 0].mean():
        aligned = aligned.copy()
        aligned[:, 0] *= -1
        aligned[:, 1] *= -1
    return aligned


# Background presets for the structures-gallery view. Each entry is
# (name, hex). Cycled by the `b` binding; passed to render_structure
# as `bg_color` so the depth-mist fades toward the actual panel bg
# rather than a hard black/gray.
_STRUCT_BG_OPTIONS = [
    ("default", None),
    ("dark",    "#0d1117"),
    ("white",   "#ffffff"),
    ("paper",   "#f4ecd8"),
]

_STATUS_STYLES = {
    "SUBMITTED": "cyan",
    "QUEUED":    "cyan",
    "RUNNING":   "yellow",
    "COMPLETED": "green",
    "FAILED":    "red",
    "CANCELLED": "dim",
    "UNKNOWN":   "dim",
}


def _format_dt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    f = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        f /= 1024.0
        if f < 1024:
            return f"{f:.1f} {unit}"
    return f"{f:.1f} PB"


def _job_status(job_id: str) -> str:
    if not _JOBS_DB.exists():
        return "?"
    try:
        with open(_JOBS_DB) as f:
            db = json.load(f)
        info = db.get(job_id) or {}
        return info.get("status", "?")
    except (json.JSONDecodeError, OSError):
        return "?"


class _StructureGalleryCanvas(Static):
    """Inner braille-rendering canvas for the structures-layer view."""

    ALLOW_SELECT = False

    def __init__(self, parent_modal: "LayerDetailModal", **kwargs) -> None:
        super().__init__("[dim]Loading…[/dim]", **kwargs)
        self._pm = parent_modal

    def render(self):
        return self._pm._render_structure_canvas(
            self.size.width, self.size.height
        )

    def on_resize(self, event) -> None:
        self.refresh()


class LayerDetailModal(ModalScreen):
    """Click a layer row -> this opens with the layer's details."""

    # Structure-gallery navigation bindings. They're listed
    # unconditionally but the action handlers no-op when the modal is
    # showing a non-structures layer, so they don't interfere with
    # other layer views.
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("left",   "struct_prev",     "←", show=False),
        Binding("right",  "struct_next",     "→", show=False),
        Binding("space",  "struct_spin",     "Spin",  show=False),
        Binding("v",      "struct_view",     "View",  show=False),
        Binding("b",      "struct_bg",       "BG",    show=False),
        Binding("f",      "struct_fetch",    "Fetch", show=False),
        Binding("m",      "struct_default",  "Make default", show=False),
    ]

    # Spin cadence — mirrors the project-detail StructureView so the
    # corner pane and this larger view rotate at the same rate.
    _STRUCT_ROTATION_FPS = 10
    _STRUCT_DEGREES_PER_FRAME = 3.0

    def __init__(self, project: BeakProject, layer_name: str) -> None:
        super().__init__()
        self._project = project
        self._layer = layer_name
        # Structures-layer state. Populated lazily on first paint of
        # the structures view; harmless for other layers.
        self._struct_cifs: List[Path] = []
        self._struct_idx: int = 0
        self._struct_coords = None
        self._struct_plddt = None
        self._struct_angle_y: float = 0.0
        self._struct_rotating: bool = True
        self._struct_error: Optional[str] = None
        self._struct_timer = None
        # Per-CIF cache populated on mount: (aligned_coords, plddt,
        # rmsd, n_matched). Aligned coords are matchmaker-superposed
        # onto the reference (cif #0); rmsd / n_matched describe the
        # quality of that fit. The reference itself has rmsd=0,
        # n_matched=len(ref). Falling back to PCA orient when
        # alignment fails (<4 matches) — better than showing the raw
        # cif frame with arbitrary translation.
        self._struct_cache: Dict[int, dict] = {}
        # View mode (`tube` = shaded ribbon w/ highlight, `trace` = wire)
        # and bg (cycles through `_STRUCT_BG_OPTIONS`). Default to the
        # Default to wire-trace mode in the gallery — matches the
        # corner pane's appearance for visual consistency, and the
        # mist + depth shading still convey 3D without the heavier
        # tube render. Tube remains a one-keypress flip away.
        self._struct_view_mode: str = "trace"
        self._struct_bg_idx: int = 0
        # Cached per-CIF rmsd-to-reference for the title. None means
        # this is the reference structure or the alignment failed.
        self._struct_rmsd: Optional[float] = None
        self._struct_n_matched: Optional[int] = None
        # Current ligands list (each: {"name", "coords"}) handed to
        # render_structure as `ligand_groups`. Empty list = no ligands
        # (typical for AlphaFold cifs).
        self._struct_ligands: List[dict] = []
        # Highlighted set name in the homologs DataTable (drives which
        # set the per-row buttons act on). None until first row paints.
        self._selected_set: str | None = None
        # Two-step delete: first click arms, second click commits.
        self._delete_armed: bool = False
        # Same two-step pattern for the embeddings-Remove button.
        # Bulk-clearing all models for a set was a footgun; the
        # row-targeted Remove still warrants a confirm because each
        # row may represent hours of compute on the remote.
        self._remove_embed_armed: bool = False
        # Per-set memo of expensive alignment metrics (mean identity,
        # Neff). The render-details path used to re-parse alignment.fasta
        # via Bio.SeqIO every time the cursor crossed a row — for a
        # 7k-sequence MSA that was 800 ms+ on the main thread, which
        # made the modal feel sluggish. Cached values are keyed by
        # (set_name, alignment_mtime) so an alignment overwrite
        # invalidates the memo automatically.
        self._aln_metrics_memo: dict[str, tuple[float, dict]] = {}
        # Cache of per-set sequence lengths for the length histogram —
        # populated by `_compute_lengths_worker` so tabbing between sets
        # only pays the parse cost on first view of each.
        self._length_cache: dict[str, list[int]] = {}
        self._length_pending: set[str] = set()
        # Per-set %ID-to-target distribution. Computed lazily from the
        # alignment.fasta when the user lands on a set with one; the
        # pane shows the length histogram first, then the identity
        # histogram once the worker finishes.
        self._identity_cache: dict[str, list[float]] = {}
        self._identity_pending: set[str] = set()
        # Set when an in-modal action mutates a set (filter, rename, …)
        # without dismissing the modal. On close we surface a marker to
        # the parent so the layers panel + views refresh.
        self._dirty: bool = False
        # Set when the structures-gallery "Make default" action writes a
        # new `view.preferred_structure`. The parent screen needs a full
        # StructureView reload (CIF swap) on close, not just a scalar
        # refresh — `_dirty` alone takes the parent down the cheap
        # `reload_set_data` path which keeps the old CIF cached.
        self._structure_default_changed: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(f"[bold]Layer · {self._layer}[/bold]", id="layer-detail-title")
            if self._layer == "homologs":
                yield from self._compose_homologs()
            elif self._layer == "embeddings":
                yield from self._compose_embeddings()
            elif self._layer == "structures":
                yield from self._compose_structures()
            else:
                yield Static(self._render_text(), id="layer-detail-content")
                with Horizontal(id="modal-buttons"):
                    yield Button("Close", id="close-btn")

    def _compose_structures(self) -> ComposeResult:
        """Gallery view: large braille-rendered structure with arrow nav.

        Replaces the old plain-text summary panel — the previous version
        listed CIF filenames + pLDDT stats but didn't actually surface
        the structure itself, which was the most useful thing about
        having one cached locally.
        """
        yield Label("", id="struct-gallery-title", classes="section-label")
        yield _StructureGalleryCanvas(self, id="struct-gallery-canvas")
        yield Label(
            "[dim]← →  switch · space  pause/resume[/dim]",
            id="struct-gallery-hint",
        )
        with Horizontal(id="modal-buttons"):
            yield Button("Close", id="close-btn")

    def _compose_homologs(self) -> ComposeResult:
        sets = self._project.homologs_sets()
        active_name = self._project.active_set_name()

        if not sets:
            yield Static(self._render_homologs(self._project.manifest()),
                         id="layer-detail-content")
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close-btn")
            return

        # Table of sets — cursor selects, Enter / row-click makes active.
        yield Label("[bold]Sets[/bold]", classes="section-label")
        table = DataTable(id="sets-table", cursor_type="row", zebra_stripes=True)
        table.add_columns(" ", "Name", "Source", "Database", "Hits", "Aligned", "Updated")
        active_row = 0
        for i, s in enumerate(sets):
            name = s.get("name", "-")
            marker = "★" if name == active_name else ""
            remote = s.get("remote") or {}
            source = s.get("source", "-")
            db = remote.get("search_database", "-")
            n = s.get("n_homologs") or 0
            n_aln = s.get("n_aligned") or 0
            aligned = f"{n_aln:,}" if n_aln else (
                "running" if remote.get("align_job_id") else "-"
            )
            updated = _format_dt(s.get("last_updated"))
            table.add_row(
                marker,
                name,
                source,
                db,
                f"{n:,}" if n else "-",
                aligned,
                updated,
                key=name,
            )
            if name == active_name:
                active_row = i
        yield table

        # Two-column row: the per-row details on the left (scrollable so
        # the modal height stays put), the length histogram on the right.
        # The histogram pane gets its own border so it reads as a chart
        # widget rather than a text continuation.
        with Horizontal(id="set-detail-row"):
            with VerticalScroll(id="set-details-scroll"):
                yield Static("", id="set-details", classes="set-details")
            yield Static("", id="length-hist-panel")

        # Action buttons grouped by intent. Each group lives in its own
        # Horizontal so a thin spacer can sit between groups without
        # collapsing the buttons together. Layout:
        #
        #   [Activate] | [Rename Filter] | [Align Taxonomy Drop align]
        #     | [Delete set]                   <spacer>     [Search] | [Close]
        #
        # — primary action on the left, edits and computes in the middle,
        # destructive on the right of the contextual group, then the
        # orthogonal "new search" + "close" pinned to the modal's right
        # edge. Tooltips still spell out the full effect on hover.
        # Three stacked rows of buttons keep the modal narrow even when
        # all 9 set actions are visible at once. Logical grouping:
        #   row 1 — set lifecycle (activate · rename · filter · dedupe)
        #   row 2 — compute / annotate (align · tax · fallback · drop align)
        #   row 3 — destructive (delete set), then a flexible spacer,
        #           then modal-level affordances (new search · close)
        # Tooltips still spell out the full effect on hover.
        # Three rows. Each row is a single Horizontal with one
        # `.btn-group` so the within-row spacing is uniform; the
        # previous design had nested groups in some rows and a flat
        # group in others, which made the columns drift visibly.
        # Search / Close on the bottom row are pushed to the right
        # edge by a 1fr spacer (modal-level actions live there by
        # convention across the rest of the app).
        # Each row has a category label on the left so the user can tell
        # at a glance what *kind* of action a button is — without it,
        # buttons read as one flat grid where Activate, Align, and
        # Delete set look like equally-weighted columnar choices. The
        # destructive `Drop align` lives in the same row as `Delete
        # set` so all data-removing actions cluster visibly together.
        with Vertical(id="modal-buttons"):
            with Horizontal(classes="btn-row"):
                yield Label("[dim]Set[/dim]", classes="row-label")
                with Horizontal(classes="btn-group"):
                    yield Button(
                        "Activate", id="make-active-btn",
                        variant="primary", compact=True,
                        tooltip=(
                            "Use this set's hits and alignment in the "
                            "project views."
                        ),
                    )
                    yield Button(
                        "Rename", id="rename-set-btn", compact=True,
                        tooltip="Rename this set.",
                    )
                    yield Button(
                        "Filter", id="filter-length-btn", compact=True,
                        tooltip=(
                            "Trim hits by sequence length — clears the "
                            "alignment for this set so you can re-align "
                            "the trimmed FASTA."
                        ),
                    )
                    yield Button(
                        "Dedupe", id="dedupe-set-btn", compact=True,
                        tooltip=(
                            "Drop exact-duplicate sequences "
                            "(case-insensitive). Clears the existing "
                            "alignment if anything is removed."
                        ),
                    )
            with Horizontal(classes="btn-row"):
                yield Label("[dim]Compute[/dim]", classes="row-label")
                with Horizontal(classes="btn-group"):
                    yield Button(
                        "Align", id="submit-align-btn", compact=True,
                        tooltip=(
                            "Submit a Clustal Omega alignment for this "
                            "set's hits. Replaces any existing alignment."
                        ),
                    )
                    yield Button(
                        "Taxonomy", id="build-tax-btn", compact=True,
                        tooltip=(
                            "Re-fetch UniProt taxonomy for this set's hits."
                        ),
                    )
                    # Visibility (and the count in the label) is wired
                    # up in `_refresh_set_actions` once the cached
                    # taxonomy parquet is on disk and we know how many
                    # rows are missing organism annotations.
                    yield Button(
                        "Resolve missing", id="tax-fallback-btn",
                        compact=True,
                        tooltip=(
                            "Run MMseqs2 LCA on the small subset of "
                            "hits UniProt couldn't resolve (typically "
                            "UniParc / metagenomic IDs). Tags resolved "
                            "rows as `mmseqs_lca` so downstream code "
                            "can weight them."
                        ),
                    )
                    yield Button(
                        "Export align", id="export-align-btn", compact=True,
                        tooltip=(
                            "Save the alignment in FASTA / Clustal / "
                            "Stockholm / PHYLIP / A2M for downstream "
                            "tools (HMMER, Jalview, tree-builders)."
                        ),
                    )
            with Horizontal(classes="btn-row"):
                yield Label("[dim]Danger[/dim]", classes="row-label")
                with Horizontal(classes="btn-group"):
                    yield Button(
                        "Drop align", id="reset-align-btn",
                        variant="warning", compact=True,
                        tooltip=(
                            "Delete the alignment but keep the hits. "
                            "Re-aligning replaces it."
                        ),
                    )
                    yield Button(
                        "Delete set", id="delete-set-btn",
                        variant="warning", compact=True,
                        tooltip=(
                            "Permanently delete this set's hits, "
                            "alignment, and metadata."
                        ),
                    )
                # 1fr spacer pushes the modal-level affordances all
                # the way to the right edge regardless of label
                # widths above — Search / Close stay anchored even
                # when the rows above grow or shrink.
                yield Static("", id="btn-spacer-grow")
                with Horizontal(classes="btn-group right-group"):
                    yield Button(
                        "Search", id="new-search-btn", compact=True,
                        tooltip=(
                            "Run a new database search — creates "
                            "another set."
                        ),
                    )
                    yield Button(
                        "Close", id="close-btn", compact=True,
                        tooltip="Close this dialog (Esc).",
                    )

        # Seed cursor on the active row so the details panel and the
        # button states match the dropdown's old default.
        self._selected_set = active_name
        self.call_after_refresh(self._select_active_row, active_row)

    def _render_text(self) -> str:
        m = self._project.manifest()
        if self._layer == "target":
            return self._render_target(m)
        if self._layer == "homologs":
            return self._render_homologs(m)
        if self._layer == "embeddings":
            return self._render_embeddings(m)
        if self._layer == "structures":
            return self._render_structures(m)
        if self._layer == "experiments":
            return self._render_experiments(m)
        return "No details for this layer."

    def _kv(self, label: str, value: str) -> str:
        return f"  {label:>14}  {value}"

    def _render_target(self, m: dict) -> str:
        t = m.get("target") or {}
        rows = []
        # Project-level context — useful at a glance.
        proj_meta = m.get("project") or {}
        if proj_meta.get("description"):
            rows.append(self._kv("Description", proj_meta["description"]))
            rows.append("")
        for label, key in (("UniProt ID", "uniprot_id"),
                           ("Gene", "gene_name"),
                           ("Organism", "organism"),
                           ("Length", "length"),
                           ("Sequence file", "sequence_file")):
            val = t.get(key)
            if val is None:
                continue
            if key == "length":
                val = f"{val} aa"
            rows.append(self._kv(label, str(val)))

        # Pfam domains (auto-populated by hmmscan).
        domains = (m.get("domains") or {}).get("hits") or []
        if domains:
            rows.append("")
            rows.append("  Pfam domains:")
            for d in domains:
                pid = d.get("pfam_id", "")
                pname = d.get("pfam_name", "")
                start = d.get("env_from", "?")
                end = d.get("env_to", "?")
                ev = d.get("i_evalue")
                ev_str = f"  e={ev:.0e}" if isinstance(ev, (int, float)) else ""
                rows.append(f"    {pid:10s}  {pname:18s}  {start}-{end}{ev_str}")

        # Structure summary if AlphaFold model is present.
        struct_dir = self._project.path / "structures"
        if struct_dir.exists():
            cifs = sorted(struct_dir.glob("*.cif"))
            if cifs:
                rows.append("")
                rows.append(self._kv("Structure", f"{cifs[0].name}"))

        return "\n".join(rows) if rows else "No target data."

    def _render_homologs(self, m: dict) -> str:
        """Fallback text rendering for the legacy code path (no sets yet)."""
        h = m.get("homologs") or {}
        rows = [self._kv("Hits", f"{h.get('n_homologs', 0):,}"
                         if h.get("n_homologs") else "-")]
        return "\n".join(rows)

    def _cached_alignment_metrics(
        self, set_name: str, aln_path: Path
    ) -> dict:
        """Return `{mean_identity, neff_at_80}` for the alignment.

        Two-tier caching:

        1. **Per-modal-instance memo** keyed on `(set_name, mtime)`:
           the cursor moving between rows in the sets table doesn't
           re-parse anything once we've seen a set this session.
        2. **Disk cache** via `alignments.cache.load_alignment_records`:
           the parsed sequences come from a numpy `.npz` sidecar
           (10s of ms) instead of a fresh BioPython parse (hundreds
           of ms to seconds for large MSAs).

        Returns `{}` when the alignment isn't on disk or the parse
        fails — callers branch on truthiness.
        """
        if not aln_path.exists():
            return {}
        try:
            mtime = aln_path.stat().st_mtime
        except OSError:
            return {}

        memo = self._aln_metrics_memo.get(set_name)
        if memo is not None and memo[0] == mtime:
            return memo[1]

        try:
            from ...alignments.cache import load_alignment_records
            from .alignment_view import (
                _mean_identity_to_target, _effective_n,
            )
            seqs = load_alignment_records(aln_path)
            if not seqs:
                return {}
            metrics = {
                "mean_identity": _mean_identity_to_target(seqs),
                "neff_at_80": _effective_n(seqs, threshold=0.8),
            }
        except Exception:
            return {}

        self._aln_metrics_memo[set_name] = (mtime, metrics)
        return metrics

    def _render_set_details(self, set_name: str) -> str:
        """Per-set details panel: runs when the table cursor moves.

        Source / Database / Hits / Aligned / Last updated are already
        in the sets table above, so we don't re-render them here —
        keeps the panel focused on what only this view shows
        (diversity metrics, taxonomy, traits). The alignment metrics
        come from a per-set memo + the npz cache, so a 7k-sequence
        MSA that used to trigger a 800ms BioPython parse on every
        cursor move now lands in tens of milliseconds on second view
        and is free on subsequent ones.
        """
        set_dict = next(
            (s for s in self._project.homologs_sets()
             if s.get("name") == set_name),
            None,
        )
        if set_dict is None:
            return ""
        remote = set_dict.get("remote") or {}
        rows: list = []
        n_aligned = set_dict.get("n_aligned", 0)

        def _section(title: str) -> None:
            """Section header — bold, separated from the prior block by
            a blank line. All subsequent rows in a section use `_kv`
            so labels right-align at column 14, giving the modal a
            single neat indent column instead of the previous mix of
            2- and 4-space ad-hoc indents."""
            if rows:
                rows.append("")
            rows.append(f"  [bold]{title}[/bold]")

        def _fmt_range(lo: float, hi: float) -> str:
            """Pick `:g` only when the value is small. Past the regular
            decimal range `:.2g` falls back to scientific notation
            (e.g. `1e+02`), which makes ranges like `23-100` render
            as `23-1e+02` and reads as garbage in the modal. Use
            integer formatting whenever the value is ≥10 and round."""
            def _one(v: float) -> str:
                av = abs(v)
                if av >= 10:
                    return f"{v:.0f}"
                if av >= 1:
                    return f"{v:.1f}"
                return f"{v:.2g}"
            return f"{_one(lo)}–{_one(hi)}"

        # Diversity metrics — only meaningful once alignment has
        # landed. Memoized + uses the .npz sidecar cache so we don't
        # re-parse the FASTA on every cursor move.
        aln_path = self._project.homologs_set_dir(set_name) / "alignment.fasta"
        metrics = self._cached_alignment_metrics(set_name, aln_path)
        if metrics:
            ident = metrics.get("mean_identity")
            neff = metrics.get("neff_at_80")
            if ident is not None or neff is not None:
                _section("Diversity")
                if ident is not None:
                    rows.append(self._kv("Mean identity", f"{ident:.0f}%"))
                if neff is not None:
                    rows.append(self._kv("Neff @80%", f"{neff} clusters"))
                if ident is not None and ident > 90:
                    rows.append(self._kv("Note", "very similar sequences (low diversity)"))
                elif neff is not None and n_aligned and neff < n_aligned * 0.1:
                    rows.append(self._kv("Note", "high redundancy"))

        # Taxonomy summary — top phyla + temperature range. Uses `·`
        # as separator (more compact than `, `) and caps phyla at 3 so
        # the line stays single-row at 60-cell modal width.
        tax_path = self._project.homologs_set_dir(set_name) / "taxonomy.parquet"
        if tax_path.exists():
            try:
                import pandas as pd
                tax = pd.read_parquet(tax_path)
                tax_rows: list = []
                if "domain" in tax.columns:
                    domain_counts = tax["domain"].dropna().value_counts().head(3)
                    if len(domain_counts):
                        tax_rows.append(self._kv(
                            "Domains",
                            " · ".join(
                                f"{d} {n}" for d, n in domain_counts.items()
                            ),
                        ))
                if "phylum" in tax.columns:
                    phylum_counts = tax["phylum"].dropna().value_counts().head(3)
                    if len(phylum_counts):
                        tax_rows.append(self._kv(
                            "Phyla",
                            " · ".join(
                                f"{p} {n}" for p, n in phylum_counts.items()
                            ),
                        ))
                if "growth_temp" in tax.columns:
                    temps = tax["growth_temp"].dropna()
                    if len(temps):
                        tax_rows.append(self._kv(
                            "Growth °C",
                            f"{_fmt_range(temps.min(), temps.max())}  "
                            f"(n={len(temps)})",
                        ))
                if tax_rows:
                    _section("Taxonomy")
                    rows.extend(tax_rows)
            except Exception:
                pass

        # metaTraits summary — only shown when the join landed.
        traits_path = self._project.homologs_set_dir(set_name) / "traits.parquet"
        if traits_path.exists():
            try:
                import pandas as pd
                tr = pd.read_parquet(traits_path)
                trait_cols = [c for c in tr.columns if c.startswith("trait_")
                              and c != "trait_match_level"]
                if trait_cols:
                    trait_rows: list = []
                    if "trait_match_level" in tr.columns:
                        n_matched = int(tr["trait_match_level"].notna().sum())
                        levels = tr["trait_match_level"].dropna().value_counts()
                        breakdown = " · ".join(
                            f"{lvl} {n}" for lvl, n in levels.items()
                        )
                        trait_rows.append(self._kv(
                            "Matched",
                            f"{n_matched}/{len(tr)}"
                            + (f"  ({breakdown})" if breakdown else ""),
                        ))
                    coverage = [
                        (c, int(tr[c].notna().sum())) for c in trait_cols
                    ]
                    coverage.sort(key=lambda kv: kv[1], reverse=True)

                    def _shorten(label: str) -> str:
                        """Trim the most-common metaTraits prefixes so
                        the displayed label fits the 14-cell kv column.
                        Bounded set of substitutions: anything not on
                        this list is left as-is (and overflows the
                        column, which is still readable, just less
                        aligned)."""
                        out = label
                        if out.startswith("presence of "):
                            out = out[len("presence of "):]
                        out = out.replace("temperature ", "temp ")
                        out = out.replace("optimum ", "opt ")
                        out = out.replace(" minimum", " min")
                        out = out.replace(" maximum", " max")
                        return out

                    shown = 0
                    for col, cov in coverage:
                        if cov == 0 or shown >= 5:
                            continue
                        s = tr[col].dropna()
                        nice = _shorten(
                            col.removeprefix("trait_").replace("_", " ")
                        )
                        if pd.api.types.is_numeric_dtype(s):
                            trait_rows.append(self._kv(
                                nice,
                                f"{_fmt_range(s.min(), s.max())}  (n={cov})",
                            ))
                        else:
                            # Drop "nan" / "None" string artefacts that
                            # leak through `.astype(str).value_counts()`.
                            top = (
                                s[~s.astype(str).str.lower().isin(
                                    ("nan", "none")
                                )]
                                .astype(str).value_counts().head(3)
                            )
                            if not len(top):
                                continue
                            trait_rows.append(self._kv(
                                nice,
                                " · ".join(
                                    f"{v} {n}" for v, n in top.items()
                                ),
                            ))
                        shown += 1
                    if trait_rows:
                        _section("Traits")
                        rows.extend(trait_rows)
            except Exception:
                pass

        # Job IDs: only show in non-terminal states. Once a job has
        # COMPLETED its IDs are noise — the Hits / Aligned columns
        # already report the outcome. Failed / running jobs warrant
        # surfacing because the user might need to click into status.
        s_jid = remote.get("search_job_id")
        a_jid = remote.get("align_job_id")
        live_jobs = []
        for label, jid in (("Search", s_jid), ("Align", a_jid)):
            if not jid:
                continue
            status = _job_status(jid)
            if status not in ("COMPLETED", "completed", "?", ""):
                live_jobs.append((label, jid, status))
        if live_jobs:
            _section("Status")
            for label, jid, status in live_jobs:
                rows.append(self._kv(f"{label} job", f"{jid} · {status}"))

        # File listing dropped — the Export button handles "I want
        # the alignment somewhere" and the table already says
        # whether each set has hits/alignment via the Aligned col.

        return "\n".join(rows)

    def _compose_embeddings(self) -> ComposeResult:
        """DataTable of (set, model) entries with per-row job status.

        Each row is one (homolog set, model) pair. Selecting a row +
        clicking Status opens the JobStatusModal for that entry's job
        — gives users a way to check on jobs that aren't currently
        the active set's most-recent (the layers-panel pill only
        surfaces the latter).
        """
        entries = self._project.embeddings_sets()

        if not entries:
            yield Static(
                "[dim]No embeddings yet — submit one with the "
                "[b]New embedding[/b] button below.[/dim]",
                id="layer-detail-content",
            )
            with Horizontal(id="modal-buttons"):
                hits_fasta = (
                    self._project.active_homologs_dir() / "sequences.fasta"
                )
                if hits_fasta.exists():
                    yield Button(
                        "New embedding",
                        id="new-embed-btn",
                        variant="primary",
                    )
                yield Button("Close", id="close-btn")
            return

        # Active set first; within a set, most-recently-updated first
        # so the row order matches what the user just submitted.
        active_name = self._project.active_set_name()

        from datetime import datetime, timezone

        def _ts(entry):
            v = entry.get("last_updated")
            if isinstance(v, datetime):
                return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).timestamp()
            try:
                return datetime.fromisoformat(str(v)).timestamp()
            except (ValueError, TypeError):
                return 0.0

        def _key(e):
            return (
                e.get("source_homologs_set") != active_name,
                -_ts(e),
            )

        entries = sorted(entries, key=_key)
        # Stash the entry list on the modal so handlers can map a
        # selected key back to its remote.job_id without re-reading
        # the manifest.
        self._embed_entries: List[Dict[str, Any]] = entries

        yield Label(
            f"[bold]Embeddings[/bold]  "
            f"[dim](active homolog set: [bold]{active_name}[/bold])[/dim]",
            classes="section-label",
        )
        table = DataTable(
            id="embeddings-table", cursor_type="row", zebra_stripes=True,
        )
        table.add_columns(
            " ", "Set", "Model", "Files", "Last updated", "Status",
        )
        from rich.text import Text
        for i, e in enumerate(entries):
            src = e.get("source_homologs_set") or "?"
            is_active = src == active_name
            # Mark + colour the Set cell so the visual association
            # "this row belongs to the active set" reads at a glance:
            # active rows render the set name in primary cyan, others
            # are dimmed grey. Combined with the section header above
            # this makes the partitioning unmistakable.
            marker = "★" if is_active else ""
            if is_active:
                src_cell = Text(src, style="bold cyan")
            else:
                src_cell = Text(src, style="dim")
            model = (e.get("model") or "-").split("/")[-1]
            n = e.get("n_embeddings") or 0
            n_str = f"{n:,}" if n else "—"
            last = _format_dt(e.get("last_updated")) or "—"
            jid = (e.get("remote") or {}).get("job_id") or ""
            status = _job_status(jid).lower() if jid else "—"
            # Row key = job id when present (for status modal lookup),
            # else a unique synthetic so duplicate (set, no-job) rows
            # don't collide.
            row_key = jid or f"row-{i}"
            table.add_row(marker, src_cell, model, n_str, last, status, key=row_key)
        yield table

        # Action row. The selected row's job_id drives `Status`; the
        # active set's hits drive `New embedding`; `Reset` always
        # operates on the active set (consistent with the previous
        # semantics and the homologs pane's bulk reset).
        with Horizontal(id="modal-buttons"):
            hits_fasta = (
                self._project.active_homologs_dir() / "sequences.fasta"
            )
            if hits_fasta.exists():
                yield Button(
                    "New embedding",
                    id="new-embed-btn",
                    variant="primary",
                )
            yield Button("Status", id="embed-status-btn")
            # Row-targeted Remove. Operates on the (set, model) pair
            # of whichever row is selected — including failed rows
            # with no `n_embeddings`. The previous bulk "Reset" wiped
            # *every* model for the active set, which made it easy to
            # accidentally trash working models alongside a failed
            # one. The button is always shown when entries exist; the
            # handler validates row selection and shows a notify if
            # nothing is highlighted.
            yield Button(
                "Remove", id="remove-embedding-btn", variant="warning",
            )
            yield Button("Close", id="close-btn")

    def _selected_embed_job_id(self) -> Optional[str]:
        """Return the job_id under the embeddings table cursor, or None."""
        try:
            table = self.query_one("#embeddings-table", DataTable)
        except Exception:
            return None
        try:
            row_key = table.coordinate_to_cell_key(
                table.cursor_coordinate
            ).row_key
        except Exception:
            return None
        key = row_key.value if row_key else None
        if not key or str(key).startswith("row-"):
            return None
        return str(key)

    def _selected_embed_entry(self) -> Optional[Dict[str, Any]]:
        """Return the embeddings entry dict under the cursor.

        Works for any row, including failed entries that have no
        `n_embeddings` and synthetic-keyed rows for entries with no
        `remote.job_id`. We map by cursor *index* into
        `self._embed_entries` so the lookup doesn't depend on the row
        key being a real job_id.
        """
        try:
            table = self.query_one("#embeddings-table", DataTable)
        except Exception:
            return None
        entries = getattr(self, "_embed_entries", None) or []
        try:
            row_idx = table.cursor_coordinate.row
        except Exception:
            return None
        if row_idx < 0 or row_idx >= len(entries):
            return None
        return entries[row_idx]

    def _render_embeddings(self, m: dict) -> str:
        # Render one block per embeddings set; the active set comes
        # first and is marked. This makes the "I have embeddings for
        # multiple homolog sets" state legible at a glance.
        sets = self._project.embeddings_sets()
        active_name = self._project.active_set_name()
        if not sets:
            return "No embeddings yet."

        def _set_sort_key(s):
            return (s.get("source_homologs_set") != active_name,
                    s.get("source_homologs_set") or "")
        sets = sorted(sets, key=_set_sort_key)

        rows = []
        for i, e in enumerate(sets):
            src = e.get("source_homologs_set") or "?"
            tag = " [dim](active)[/dim]" if src == active_name else ""
            if i > 0:
                rows.append("")
            rows.append(f"[bold]Set: {src}[/bold]{tag}")
            remote = e.get("remote") or {}
            rows.append(self._kv("Model", e.get("model", "-") or "-"))
            n = e.get("n_embeddings", 0)
            rows.append(self._kv("Files", f"{n:,}" if n else "-"))
            last = e.get("last_updated")
            if last is not None:
                rows.append(self._kv("Last updated", _format_dt(last)))
            jid = remote.get("job_id")
            if jid:
                rows.append(self._kv("Job", f"{jid}  -  {_job_status(jid)}"))

            set_dir = self._project.embeddings_set_dir(src)
            if set_dir.exists():
                files = sorted(p for p in set_dir.rglob("*") if p.is_file())[:6]
                if files:
                    rows.append("  Sample files:")
                    for p in files:
                        rows.append(
                            f"    - {p.relative_to(set_dir)}  "
                            f"({_human_size(p.stat().st_size)})"
                        )
        return "\n".join(rows)

    def _render_structures(self, m: dict) -> str:
        d = self._project.path / "structures"
        if not d.exists():
            return "No structures yet."
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            return "No structures yet."
        rows = [self._kv("Files", str(len(cifs)))]

        # Pull pLDDT + SS stats from the first CIF.
        cif = cifs[0]
        try:
            from ..structure import load_ca_coords, parse_secondary_structure
            _, plddt = load_ca_coords(cif)
            n_residues = len(plddt)
            rows.append(self._kv("Residues", f"{n_residues}"))
            rows.append(self._kv(
                "pLDDT range", f"{plddt.min():.0f} – {plddt.max():.0f}"
            ))
            rows.append(self._kv("Mean pLDDT", f"{plddt.mean():.0f}"))
            ss = parse_secondary_structure(cif, n_residues)
            if ss:
                n_h = ss.count("H")
                n_e = ss.count("E")
                n_c = n_residues - n_h - n_e
                rows.append(self._kv(
                    "Secondary structure",
                    f"H {n_h * 100 // max(1, n_residues)}%  "
                    f"E {n_e * 100 // max(1, n_residues)}%  "
                    f"loop {n_c * 100 // max(1, n_residues)}%",
                ))
        except Exception:
            pass

        rows.append("")
        for p in cifs:
            rows.append(f"  - {p.name}  ({_human_size(p.stat().st_size)})")
        return "\n".join(rows)

    def _render_experiments(self, m: dict) -> str:
        exps = m.get("experiments") or []
        if not exps:
            return "No experiments imported yet."
        rows = [self._kv("Count", str(len(exps))), ""]
        for e in exps:
            name = e.get("name") if isinstance(e, dict) else str(e)
            rows.append(f"  - {name}")
        return "\n".join(rows)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        # Any button click other than the delete arming itself disarms it.
        if bid != "delete-set-btn" and self._delete_armed:
            self._disarm_delete()
        # Same disarm pattern for the embeddings-Remove confirm — any
        # other button click cancels the pending remove.
        if bid != "remove-embedding-btn" and self._remove_embed_armed:
            self._disarm_remove_embed()
        if bid == "close-btn":
            self.dismiss(self._close_action())
        elif bid == "reset-align-btn":
            self._reset_alignment_for(self._selected_set)
        elif bid == "reset-homologs-btn":
            self._reset_homologs()
        elif bid == "remove-embedding-btn":
            self._remove_embedding_clicked(event.button)
        elif bid == "build-tax-btn":
            self._build_taxonomy_now(event.button)
        elif bid == "tax-fallback-btn":
            self._submit_tax_fallback(event.button)
        elif bid == "make-active-btn":
            self._make_selected_active()
        elif bid == "delete-set-btn":
            self._delete_set_clicked(event.button)
        elif bid == "new-search-btn":
            self._open_new_search()
        elif bid == "new-embed-btn":
            self.dismiss("open-embed")
        elif bid == "embed-status-btn":
            self._open_embed_status()
        elif bid == "rename-set-btn":
            self._open_rename_set()
        elif bid == "filter-length-btn":
            self._open_filter_length()
        elif bid == "dedupe-set-btn":
            self._open_dedupe_set()
        elif bid == "submit-align-btn":
            self._submit_align_clicked()
        elif bid == "export-align-btn":
            self._open_export_alignment()

    def _open_embed_status(self) -> None:
        """Open JobStatusModal for the embeddings row under the cursor."""
        jid = self._selected_embed_job_id()
        if not jid:
            self.notify(
                "Select a row with a tracked job (status column ≠ —).",
                timeout=4,
            )
            return
        from .job_status import JobStatusModal
        self.app.push_screen(JobStatusModal(jid))

    def _open_new_search(self) -> None:
        """Dismiss with a marker so the detail screen opens the search modal."""
        self.dismiss("open-search")

    def _open_export_alignment(self) -> None:
        """Open the format/path picker for the selected set's alignment.

        Disabled-state behavior is enforced here rather than via
        `Button.disabled` because the active set can change between
        modal-mount and click — a worker that just landed an alignment
        flips the eligibility on. Keeping the gate dynamic means the
        button is "always clickable, sometimes complains" instead of
        "sometimes hidden, sometimes broken when stale."
        """
        if not self._selected_set:
            return
        aln_path = (
            self._project.homologs_set_dir(self._selected_set)
            / "alignment.fasta"
        )
        if not aln_path.exists():
            self.notify(
                f"No alignment for '{self._selected_set}' — run Align "
                "first.",
                severity="warning", timeout=6,
            )
            return
        from .export_alignment import ExportAlignmentModal
        # Suggested filename: `<project>_<set>_alignment` so files
        # exported from multiple sets / projects don't collide in the
        # download folder.
        stem = f"{self._project.name}_{self._selected_set}_alignment"
        self.app.push_screen(
            ExportAlignmentModal(aln_path, default_stem=stem),
            self._on_export_done,
        )

    def _on_export_done(self, out_path) -> None:
        if not out_path:
            return
        self.notify(
            f"Exported to {out_path}",
            timeout=6,
        )

    def _submit_align_clicked(self) -> None:
        """Hand the parent screen the (re-)align request for the highlighted set.

        Reuses `LayersPanel.submit_alignment(set_name)` over there, which
        owns the SSH manager + worker + status pill — rather than spawning
        a parallel submit path inside the modal. The marker carries the
        set name so the parent can target the right one even when it
        isn't the active set.
        """
        if not self._selected_set:
            return
        # Cheap sanity: there has to be a hits FASTA to align.
        hits = self._project.homologs_set_dir(self._selected_set) / "sequences.fasta"
        if not hits.exists():
            self.notify(
                f"No sequences.fasta in '{self._selected_set}' — "
                "search first.", severity="warning", timeout=6,
            )
            return
        self.dismiss(f"submit-align:{self._selected_set}")

    def _open_dedupe_set(self) -> None:
        """Push the dedupe confirmation modal for the highlighted set."""
        if not self._selected_set:
            return
        set_name = self._selected_set
        from .dedupe_set import DedupeSetModal

        def _on_deduped(result) -> None:
            if not result:
                return
            n_kept, n_dropped = result
            if n_dropped == 0:
                self.notify(
                    f"'{set_name}' had no exact duplicates "
                    f"({n_kept:,} sequences).",
                    timeout=6,
                )
                return
            # Mirror the filter path: cached lengths and any in-flight
            # length parse are stale; mark the modal dirty so the
            # parent refreshes layers/views on close. Same goes for
            # the identity cache — dedup changes the homolog membership
            # so any cached %ID distribution is no longer accurate.
            self._length_cache.pop(set_name, None)
            self._length_pending.discard(set_name)
            self._identity_cache.pop(set_name, None)
            self._identity_pending.discard(set_name)
            self._refresh_set_actions()
            self._dirty = True
            self.notify(
                f"Removed {n_dropped:,} duplicate(s) from '{set_name}' · "
                f"{n_kept:,} unique sequences remain.",
                timeout=6,
            )

        self.app.push_screen(DedupeSetModal(self._project, set_name), _on_deduped)

    def _open_filter_length(self) -> None:
        """Push the length-filter modal for the highlighted set."""
        if not self._selected_set:
            return
        set_name = self._selected_set
        # Seed the inputs with the set's actual length range when we
        # know it; fall back to a wide default so the user always has
        # something sensible to edit.
        lengths = self._length_cache.get(set_name) or []
        cur_min = min(lengths) if lengths else 0
        cur_max = max(lengths) if lengths else 1000
        from .filter_length import FilterLengthModal

        def _on_filtered(result) -> None:
            if not result:
                return
            mn, mx, n_kept = result
            # Invalidate cached lengths + identities so the histograms
            # re-parse the trimmed FASTA / re-aligned alignment on the
            # next refresh.
            self._length_cache.pop(set_name, None)
            self._length_pending.discard(set_name)
            self._identity_cache.pop(set_name, None)
            self._identity_pending.discard(set_name)
            self._refresh_set_actions()
            # Mark dirty so the parent screen refreshes when the modal
            # closes — the layers panel needs to pick up the new
            # n_homologs and the cleared alignment.
            self._dirty = True
            self.notify(
                f"Filtered '{set_name}' to {n_kept:,} sequences in "
                f"[{mn}, {mx}] aa.",
                timeout=6,
            )

        self.app.push_screen(
            FilterLengthModal(self._project, set_name, cur_min, cur_max),
            _on_filtered,
        )

    def _open_rename_set(self) -> None:
        """Push the rename modal for the highlighted set; refresh on success."""
        if not self._selected_set:
            return
        old_name = self._selected_set
        from .rename_set import RenameSetModal

        def _on_renamed(new_name: str | None) -> None:
            if not new_name or new_name == old_name:
                return
            # Re-seed the table cursor on the renamed row. Cheapest path
            # is to dismiss with a marker — the parent screen already
            # rebuilds the layers panel for `set-switched`, and the
            # active set's name may have changed if we renamed it.
            self.dismiss("set-renamed")

        self.app.push_screen(RenameSetModal(self._project, old_name), _on_renamed)

    # ---- homologs sets table interactions ----

    def _select_active_row(self, row_idx: int) -> None:
        try:
            table = self.query_one("#sets-table", DataTable)
        except Exception:
            return
        # Clamp into the current row range so a stale `row_idx` (e.g.
        # from a manifest read between compose and call_after_refresh)
        # can't trip move_cursor on an out-of-bounds row.
        n_rows = max(0, table.row_count)
        if n_rows == 0:
            return
        row_idx = max(0, min(row_idx, n_rows - 1))
        try:
            table.move_cursor(row=row_idx, animate=False)
        except Exception:
            pass
        self._refresh_set_actions()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "sets-table":
            return
        key = event.row_key.value if event.row_key else None
        self._selected_set = str(key) if key else None
        # Switching rows resets the delete confirmation.
        if self._delete_armed:
            self._disarm_delete()
        self._refresh_set_actions()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter / row-click on a non-active row promotes it to active.
        if event.data_table.id != "sets-table":
            return
        key = event.row_key.value if event.row_key else None
        if not key:
            return
        if str(key) == self._project.active_set_name():
            return
        self._selected_set = str(key)
        self._make_selected_active()

    def _refresh_set_actions(self) -> None:
        """Sync the per-row details panel + button visibility/state."""
        try:
            details = self.query_one("#set-details", Static)
        except Exception:
            return
        if not self._selected_set:
            details.update("")
            self._update_length_panel("")
            return
        details.update(self._render_set_details(self._selected_set))
        self._refresh_length_panel(self._selected_set)

        active = self._project.active_set_name()
        sets = self._project.homologs_sets()
        set_dict = next((s for s in sets if s.get("name") == self._selected_set), {})
        is_active = self._selected_set == active

        try:
            make_active_btn = self.query_one("#make-active-btn", Button)
            make_active_btn.disabled = is_active
        except Exception:
            pass

        # Disable Export when the selected set has no alignment yet —
        # the click handler also guards, but disabling here removes
        # the need to "try and see" for the user.
        try:
            export_btn = self.query_one("#export-align-btn", Button)
            aln_path = (
                self._project.homologs_set_dir(self._selected_set)
                / "alignment.fasta"
            )
            export_btn.disabled = not aln_path.exists()
        except Exception:
            pass

        try:
            tax_btn = self.query_one("#build-tax-btn", Button)
            tax_path = (
                self._project.homologs_set_dir(self._selected_set)
                / "taxonomy.parquet"
            )
            has_hits = bool(set_dict.get("n_homologs"))
            # Stay enabled when the cached parquet is *incomplete*
            # — i.e. the FASTA has seq_ids the parquet hasn't seen
            # (e.g. user re-ran search and homologs grew from 80 →
            # 378). The build is incremental: it'll fetch only the
            # uncovered IDs and merge them into the existing parquet.
            uncovered = self._tax_uncovered_count_for(self._selected_set)
            fully_covered = tax_path.exists() and uncovered == 0
            tax_btn.disabled = not has_hits or fully_covered
            if has_hits and uncovered > 0 and tax_path.exists():
                tax_btn.label = f"Resolve {uncovered} new"
            else:
                tax_btn.label = "Resolve"
        except Exception:
            pass

        # Fallback button: only relevant when taxonomy.parquet is on
        # disk AND has unresolved rows. Hidden otherwise so the row
        # doesn't suggest work that isn't applicable.
        try:
            fb_btn = self.query_one("#tax-fallback-btn", Button)
            self._tax_fallback_unresolved = self._unresolved_count_for(
                self._selected_set
            )
            n = self._tax_fallback_unresolved
            if n > 0:
                fb_btn.label = f"Resolve missing ({n})"
                fb_btn.display = True
            else:
                fb_btn.display = False
        except Exception:
            pass

        try:
            align_btn = self.query_one("#reset-align-btn", Button)
            align_btn.disabled = not bool(set_dict.get("n_aligned"))
        except Exception:
            pass

        # The "Align" / "Re-align" button — enabled when:
        #   - the set has hits to align,
        #   - no alignment job is currently in flight for it.
        # Label flips to "Re-align" when an alignment already exists, so
        # the destructive intent is obvious.
        try:
            submit_align_btn = self.query_one("#submit-align-btn", Button)
            has_hits = bool(set_dict.get("n_homologs"))
            in_flight = bool((set_dict.get("remote") or {}).get("align_job_id"))
            already_aligned = bool(set_dict.get("n_aligned"))
            submit_align_btn.disabled = not has_hits or in_flight
            submit_align_btn.label = "Re-align" if already_aligned else "Align"
        except Exception:
            pass

        try:
            del_btn = self.query_one("#delete-set-btn", Button)
            # Always allow delete (the project tolerates losing all sets);
            # arming state is owned by `_delete_set_clicked`.
            del_btn.disabled = False
        except Exception:
            pass

        try:
            rename_btn = self.query_one("#rename-set-btn", Button)
            # Rename always available once a row is highlighted — the
            # modal validates the new name itself.
            rename_btn.disabled = self._selected_set is None
        except Exception:
            pass

    def _unresolved_count_for(self, set_name: Optional[str]) -> int:
        """How many hits in `set_name`'s taxonomy.parquet have no organism."""
        if not set_name:
            return 0
        cache = self._project.homologs_set_dir(set_name) / "taxonomy.parquet"
        if not cache.exists():
            return 0
        try:
            import pandas as pd
            df = pd.read_parquet(cache)
        except Exception:
            return 0
        if "organism" not in df.columns:
            return 0
        return int(df["organism"].isna().sum())

    def _tax_uncovered_count_for(self, set_name: Optional[str]) -> int:
        """How many FASTA seq_ids are absent from the taxonomy parquet.

        Returns 0 when the parquet doesn't exist (covered = nothing
        because nothing has been built yet — caller distinguishes via
        `tax_path.exists()`). Used by the Resolve button to detect a
        stale cache that grew under it.
        """
        if not set_name:
            return 0
        hd = self._project.homologs_set_dir(set_name)
        cache = hd / "taxonomy.parquet"
        fasta = hd / "sequences.fasta"
        if not cache.exists() or not fasta.exists():
            return 0
        try:
            import pandas as pd
            df = pd.read_parquet(cache)
        except Exception:
            return 0
        if "sequence_id" not in df.columns:
            return 0
        cached_ids = set(df["sequence_id"].astype(str))
        from ..taxonomy import _parse_fasta_ids
        fasta_ids = _parse_fasta_ids(fasta)
        return sum(1 for s in fasta_ids if s not in cached_ids)

    def _make_selected_active(self) -> None:
        if not self._selected_set:
            return
        if self._project.set_active_set(self._selected_set):
            self.dismiss("set-switched")

    def _delete_set_clicked(self, button: Button) -> None:
        """Two-step delete: first click arms the button, second commits."""
        if not self._selected_set:
            return
        if not self._delete_armed:
            self._delete_armed = True
            button.label = f"Confirm delete '{self._selected_set}'?"
            return
        # Commit the delete.
        name = self._selected_set
        self._delete_armed = False
        if self._project.delete_homolog_set(name):
            self.notify(f"Deleted set '{name}'", timeout=4)
            self.dismiss("set-deleted")

    def _disarm_delete(self) -> None:
        self._delete_armed = False
        try:
            self.query_one("#delete-set-btn", Button).label = "Delete set"
        except Exception:
            pass

    def _build_taxonomy_now(self, button) -> None:
        """Trigger an on-demand UniProt-REST taxonomy build for the active set."""
        button.disabled = True
        button.label = "Building…"
        self._taxonomy_worker()

    def _submit_tax_fallback(self, button) -> None:
        """Submit MMseqs LCA on the unresolved subset of the active set.

        Writes a `taxonomy_fallback.fasta` containing just the rows
        that UniProt couldn't resolve, then submits a small MMseqs
        taxonomy job via the existing `MMseqsTaxonomy` infra. The
        existing pull worker (`_pull_taxonomy_now` in layers_panel)
        invokes `merge_mmseqs_fallback` on completion to upsert the
        canonical parquet.
        """
        n = getattr(self, "_tax_fallback_unresolved", 0)
        if not n:
            return
        if not self._selected_set:
            return
        button.disabled = True
        button.label = f"Submitting ({n})…"
        self._tax_fallback_worker(self._selected_set)

    # Preference order for the fallback database — smallest viable DB
    # that has taxonomy info wins. SwissProt is ~570k sequences (LCA
    # in seconds); UniRef50 is ~50M (a minute or two); UniRef90/100
    # and UniProtKB only get used if nothing smaller is taxonomy-
    # enabled on this workstation.
    _TAX_FALLBACK_DB_PREFS = (
        "swissprot", "uniref50", "uniprotkb", "uniref90", "uniref100",
    )

    @work(thread=True, exclusive=True, group="tax-fallback")
    def _tax_fallback_worker(self, set_name: str) -> None:
        from ..taxonomy import unresolved_seq_ids, write_fallback_fasta

        try:
            ids = unresolved_seq_ids(self._project)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Fallback aborted: {e}",
                severity="error", timeout=8,
            )
            return
        if not ids:
            self.app.call_from_thread(
                self.notify, "No unresolved sequences to backfill.",
                timeout=4,
            )
            return

        fasta = write_fallback_fasta(self._project, ids)
        if fasta is None:
            self.app.call_from_thread(
                self.notify, "Couldn't stage fallback FASTA "
                "(no matching headers).",
                severity="error", timeout=8,
            )
            return

        mgr = None
        try:
            from ...remote.taxonomy import MMseqsTaxonomy
            mgr = MMseqsTaxonomy()

            # Probe for taxonomy-enabled DBs once and pick the smallest
            # we find. Saves a round-trip per failed submit attempt and
            # surfaces a clean error when none are prepared.
            chosen_db = self._pick_fallback_db(mgr)
            if chosen_db is None:
                self.app.call_from_thread(
                    self.notify,
                    "No taxonomy-enabled DB on the remote. Ask the admin "
                    "to run `mmseqs createtaxdb <db> tmp` against one of: "
                    + ", ".join(self._TAX_FALLBACK_DB_PREFS),
                    severity="error", timeout=12,
                )
                return

            job_name = f"{self._project.name}_{set_name}_taxfallback"
            job_id = mgr.submit(
                str(fasta), database=chosen_db, job_name=job_name,
            )
            with self._project.mutate() as m:
                tax = m.setdefault("taxonomy", {})
                remote = tax.setdefault("remote", {})
                remote["job_id"] = job_id
                remote["database"] = chosen_db
                tax["fallback_n_seqs"] = len(ids)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Fallback submit failed: {e}",
                severity="error", timeout=10,
            )
            return
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass

        self.app.call_from_thread(
            self.notify,
            f"MMseqs LCA fallback submitted ({chosen_db}) for "
            f"{len(ids)} unresolved sequence(s) · {job_id}",
            timeout=8,
        )
        # Mark dirty so the parent screen refreshes the layers panel
        # on dismiss — the new tax job_id will animate the homologs
        # row's marquee with `⊳ taxonomy` until it lands.
        self._dirty = True
        self.app.call_from_thread(self._refresh_set_actions)

    def _pick_fallback_db(self, mgr) -> Optional[str]:
        """Return the smallest taxonomy-enabled DB alias on the remote.

        Walks `_TAX_FALLBACK_DB_PREFS` in order, asking the remote
        whether each `<db>_taxonomy` sibling file exists. Returns the
        first match — `mgr.submit` will reject a DB without taxonomy
        info anyway, so this avoids a round-trip per failed attempt
        and lets the worker tell the user *which* DB it picked.
        """
        try:
            for alias in self._TAX_FALLBACK_DB_PREFS:
                db_name = mgr.AVAILABLE_DBS.get(alias, alias)
                db_path = f"{mgr.DB_BASE_PATH}/{db_name}"
                check = mgr.conn.run(
                    f'[ -e {db_path} ] && [ -e {db_path}_taxonomy ] '
                    f'&& echo OK || echo NO',
                    hide=True, warn=True,
                )
                if check.ok and check.stdout.strip() == "OK":
                    return alias
        except Exception:  # noqa: BLE001
            pass
        return None

    @work(thread=True, exclusive=True, group="tax-rebuild")
    def _taxonomy_worker(self) -> None:
        def _on_progress(done: int, total: int) -> None:
            # Live-update the button label so the user can see the
            # build progressing through several minutes of UniProt
            # round-trips on a 25k+-hit set.
            try:
                self.app.call_from_thread(
                    self._update_tax_button_progress, done, total
                )
            except Exception:
                pass

        try:
            from ..taxonomy import build_taxonomy_table
            df = build_taxonomy_table(
                self._project, force=True, progress_cb=_on_progress
            )
            n = len(df) if df is not None else 0
            n_resolved = (
                int(df["organism"].notna().sum())
                if df is not None and "organism" in df.columns else 0
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Taxonomy build failed: {e}",
                severity="error", timeout=8,
            )
            return
        self.app.call_from_thread(
            self.notify,
            f"Taxonomy: {n_resolved}/{n} sequences", timeout=6,
        )
        # Dismiss with a marker the detail screen will recognize so the
        # layers panel refreshes the homolog row.
        self.app.call_from_thread(self.dismiss, "tax-rebuilt")

    def _update_tax_button_progress(self, done: int, total: int) -> None:
        """UI-thread label update for the Taxonomy button."""
        try:
            btn = self.query_one("#build-tax-btn", Button)
        except Exception:
            return
        if total <= 0:
            btn.label = "Building…"
            return
        pct = int(min(100, max(0, done / total * 100))) if total else 0
        btn.label = f"{done:,}/{total:,} · {pct}%"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "active-set-select":
            new_name = str(event.value)
            if new_name and new_name != self._project.active_set_name():
                self._project.set_active_set(new_name)
                self.dismiss("set-switched")

    def action_close(self) -> None:
        self.dismiss(self._close_action())

    def _close_action(self) -> Optional[str]:
        """Pick the most specific dismiss marker for the parent screen.

        `structure-default-changed` requires a full CIF reload, while
        `set-mutated` only needs a scalar refresh — surface the heavier
        signal whenever both flags fire so the parent doesn't take the
        cheap path and leave the old structure on screen.
        """
        if self._structure_default_changed:
            return "structure-default-changed"
        if self._dirty:
            return "set-mutated"
        return None

    # ---- structures-layer gallery ----

    def on_mount(self) -> None:
        # Lazy-init the gallery only when we're showing the structures
        # layer. Other layers ignore the spin tick + state entirely.
        if self._layer != "structures":
            return
        self._struct_cifs = sorted(
            (self._project.path / "structures").glob("*.cif")
        )
        # Build the matchmaker-aligned cache up front. This is one
        # pairwise alignment + one Kabsch per non-reference CIF; for
        # the typical 1-3 structures per project it's <50 ms total.
        # Larger sets (10+) might want a worker, but keep it simple
        # until that's a real concern.
        self._struct_build_cache()
        self._struct_load_current()
        self._struct_timer = self.set_interval(
            1.0 / self._STRUCT_ROTATION_FPS, self._struct_tick,
        )

    def _struct_build_cache(self) -> None:
        """Pre-load + matchmaker-align every CIF against the first.

        Reference is `cifs[0]`. For each other CIF: pairwise sequence
        align to the reference, Kabsch superpose on matched CA pairs.
        Failed alignments (<4 matches) fall back to PCA-canonical
        orientation, which still puts each structure into a comparable
        frame even when sequence alignment can't anchor it.
        """
        from ..structure import load_ca_residues
        if not self._struct_cifs:
            return

        import numpy as np
        from ..structure import load_ligand_atoms

        # Reference: load + center on its own centroid so the overall
        # cloud is at the origin. PCA-canonical for the reference too —
        # this is what every superposed structure inherits the frame of.
        ref_path = self._struct_cifs[0]
        try:
            ref_coords, ref_plddt, _, ref_seq = load_ca_residues(ref_path)
            ref_oriented = _pca_canonical_orient(ref_coords)
        except Exception as e:  # noqa: BLE001
            self._struct_cache[0] = {
                "coords": None, "plddt": None,
                "rmsd": None, "n_matched": None,
                "error": f"reference load failed: {e}",
                "seq": "", "ligands": [], "n_lig_atoms": 0,
            }
            return

        # The Kabsch superposition runs in raw cif coords first, then
        # the reference's centering + PCA rotation is applied to land
        # the result in the same canonical frame the user sees the
        # reference in. Capture (centroid, R_ref) once — we apply the
        # same transform to ligand atoms so they stay attached to
        # the protein through the orientation pipeline.
        ref_centroid = ref_coords.mean(axis=0)
        ref_centered = ref_coords - ref_centroid
        R_ref, _, _, _ = np.linalg.lstsq(ref_centered, ref_oriented, rcond=None)

        def _apply_ref_frame(ligs):
            """Translate by -ref_centroid then rotate by R_ref."""
            out = []
            for g in ligs:
                c = g.get("coords")
                if c is None or len(c) == 0:
                    continue
                out.append({
                    "name": g.get("name", "?"),
                    "coords": (c - ref_centroid) @ R_ref,
                })
            return out

        # Ligands: pulled once per CIF, transformed into the canonical
        # frame so they ride along with their parent protein.
        try:
            ref_ligs_raw = load_ligand_atoms(ref_path)
        except Exception:
            ref_ligs_raw = []
        ref_ligs = _apply_ref_frame(ref_ligs_raw)

        self._struct_cache[0] = {
            "coords": ref_oriented,
            "plddt": ref_plddt,
            "rmsd": 0.0,
            "n_matched": len(ref_oriented),
            "error": None,
            "seq": ref_seq,
            "ligands": ref_ligs,
            "n_lig_atoms": sum(len(g["coords"]) for g in ref_ligs),
        }

        for i in range(1, len(self._struct_cifs)):
            path = self._struct_cifs[i]
            try:
                coords, plddt, _, seq = load_ca_residues(path)
            except Exception as e:  # noqa: BLE001
                self._struct_cache[i] = {
                    "coords": None, "plddt": None,
                    "rmsd": None, "n_matched": None,
                    "error": f"load failed: {e}", "seq": "",
                    "ligands": [], "n_lig_atoms": 0,
                }
                continue

            # Per-CIF ligands in the cif's raw frame. We'll transform
            # them with the same Kabsch + ref-frame applied to the
            # protein so they stay docked.
            try:
                mob_ligs_raw = load_ligand_atoms(path)
            except Exception:
                mob_ligs_raw = []

            # Step 1: superpose mob protein onto reference's *raw* frame.
            # `_matchmaker_superpose` now returns (R_kab, t_kab) so we
            # can apply the *exact same* rigid transform to ligands —
            # the previous code recovered an effective rotation via
            # lstsq, which absorbed translation into a non-orthogonal
            # matrix and silently displaced bound ligands away from
            # their host protein.
            aligned_to_raw_ref, rmsd, R_kab, t_kab = _matchmaker_superpose(
                ref_coords, ref_seq, coords, seq,
            )
            n_matched = None
            mob_ligs_aligned: list = []

            if aligned_to_raw_ref is None:
                # Sequence alignment too poor — PCA canonical fallback.
                # Without per-protein matchmaker we can't superpose
                # ligands meaningfully either, so we skip them.
                aligned_in_canonical = _pca_canonical_orient(coords)
                rmsd = None
            else:
                # Apply ref-centroid + ref-frame rotation to land in
                # the same canonical frame as ref_oriented.
                aligned_in_canonical = (
                    (aligned_to_raw_ref - ref_centroid) @ R_ref
                )

                # Ligands ride through the same pipeline as the
                # protein: cif coords → Kabsch (R_kab, t_kab) → ref
                # frame. Mathematically: aligned_lig = c @ R_kab.T +
                # t_kab, then (aligned_lig - ref_centroid) @ R_ref.
                for g in mob_ligs_raw:
                    c = g.get("coords")
                    if c is None or len(c) == 0:
                        continue
                    aligned_lig = c @ R_kab.T + t_kab
                    final_lig = (aligned_lig - ref_centroid) @ R_ref
                    mob_ligs_aligned.append({
                        "name": g.get("name", "?"),
                        "coords": final_lig,
                    })

                # Recover n_matched from the aligner. Cheaper to
                # recompute than to thread back through the helper.
                try:
                    from Bio.Align import PairwiseAligner, substitution_matrices
                    a = PairwiseAligner()
                    a.mode = "global"
                    try:
                        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
                    except Exception:
                        a.match_score = 2
                        a.mismatch_score = -1
                    a.open_gap_score = -10
                    a.extend_gap_score = -1
                    first = a.align(ref_seq, seq)[0]
                    n_matched = sum(
                        min(re - rs, me - ms)
                        for (rs, re), (ms, me) in zip(*first.aligned)
                    )
                except Exception:
                    n_matched = None

            self._struct_cache[i] = {
                "coords": aligned_in_canonical,
                "plddt": plddt,
                "rmsd": rmsd,
                "n_matched": n_matched,
                "error": None,
                "seq": seq,
                "ligands": mob_ligs_aligned,
                "n_lig_atoms": sum(
                    len(g["coords"]) for g in mob_ligs_aligned
                ),
            }

    def action_struct_prev(self) -> None:
        if self._layer != "structures" or len(self._struct_cifs) <= 1:
            return
        self._struct_idx = (self._struct_idx - 1) % len(self._struct_cifs)
        self._struct_load_current()

    def action_struct_next(self) -> None:
        if self._layer != "structures" or len(self._struct_cifs) <= 1:
            return
        self._struct_idx = (self._struct_idx + 1) % len(self._struct_cifs)
        self._struct_load_current()

    def action_struct_spin(self) -> None:
        if self._layer != "structures":
            return
        self._struct_rotating = not self._struct_rotating

    def action_struct_view(self) -> None:
        """Toggle between shaded tube and wire trace."""
        if self._layer != "structures":
            return
        self._struct_view_mode = (
            "trace" if self._struct_view_mode == "tube" else "tube"
        )
        self._struct_update_title()
        self._struct_refresh_canvas()

    def action_struct_fetch(self) -> None:
        """Query SIFTS for alternate PDBs and download missing ones.

        Best-effort: pulls top-3 structures for the project's UniProt
        ID via PDBe SIFTS. Skips ones already on disk (filename match).
        On completion, rescans the structures dir and rebuilds the
        cache so the new CIFs become switchable with ←/→.
        """
        if self._layer != "structures":
            return
        target = (self._project.manifest().get("target") or {})
        uniprot = target.get("uniprot_id")
        if not uniprot:
            self.notify(
                "Fetch needs a UniProt ID on the target — none recorded.",
                severity="warning", timeout=6,
            )
            return
        self.notify(f"Fetching alternate PDBs for {uniprot}…", timeout=4)
        self._struct_fetch_worker(uniprot)

    @work(thread=True, exclusive=True, group="struct-fetch")
    def _struct_fetch_worker(self, uniprot: str) -> None:
        try:
            from ...api.structures import find_structures, fetch_structures
            # Pull both sources by default: top-3 experimental PDBs +
            # the AlphaFold prediction. AF is always informative — it
            # covers full-length even when every PDB is partial, and
            # the gallery's `n` keybind lets the user flip between
            # them. Costs ~one extra ~500 KB download per project.
            df = find_structures([uniprot], source="both")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"SIFTS lookup failed: {e}",
                severity="error", timeout=8,
            )
            return
        dest = self._project.path / "structures"
        if df is None or df.empty:
            self.app.call_from_thread(
                self.notify,
                f"No structures found for {uniprot}.",
                timeout=6,
            )
            return
        try:
            # Hand-pick top 3 PDBs *and* the single AF row. The
            # `_select_structures("top N")` ranker would otherwise
            # crowd AF out by sorting on resolution (AF rows have
            # `resolution=None` → ranked last), but we want AF cached
            # always so the user can flip to a confidence-bands view
            # of full-length coverage even when every PDB is partial.
            pdb_rows = df[df["source"] == "pdb"].head(3)
            af_rows = df[df["source"] == "alphafold"].head(1)
            import pandas as _pd
            picked = _pd.concat([pdb_rows, af_rows], ignore_index=True)
            fetched = fetch_structures(
                picked, output_dir=str(dest),
                selection="all",  # we already filtered above
                skip_existing=True,
            )
            ok = fetched[fetched["local_path"].notna()]
            n_new = len(ok)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Fetch failed: {e}",
                severity="error", timeout=8,
            )
            return
        # Rebuild gallery cache on the UI thread so the new CIFs
        # become navigable.
        self.app.call_from_thread(self._struct_rescan_after_fetch, n_new)

    def _struct_rescan_after_fetch(self, n_new: int) -> None:
        self._struct_cifs = sorted(
            (self._project.path / "structures").glob("*.cif")
        )
        self._struct_cache.clear()
        self._struct_idx = min(self._struct_idx, len(self._struct_cifs) - 1)
        self._struct_build_cache()
        self._struct_load_current()
        self.notify(
            f"Fetched {n_new} new structure(s); now {len(self._struct_cifs)} cached.",
            timeout=6,
        )

    def action_struct_default(self) -> None:
        """Mark the current CIF as the project's preferred structure.

        Records the basename in the manifest under ``[view]
        preferred_structure``. The project-detail screen's
        StructureView consults this on load so subsequent opens of the
        project use the chosen file instead of the alphabetic-first
        PDB cached in the structures dir.
        """
        if self._layer != "structures" or not self._struct_cifs:
            return
        chosen = self._struct_cifs[self._struct_idx].name
        try:
            with self._project.mutate() as m:
                view = m.setdefault("view", {})
                view["preferred_structure"] = chosen
        except Exception as e:  # noqa: BLE001
            self.notify(
                f"Couldn't write preference: {e}",
                severity="error", timeout=6,
            )
            return
        self._dirty = True  # parent screen refreshes the StructureView on close
        self._structure_default_changed = True
        self.notify(
            f"Set {chosen} as default structure for this project.",
            timeout=6,
        )

    def action_struct_bg(self) -> None:
        """Cycle through the bg presets (default → dark → white → paper)."""
        if self._layer != "structures":
            return
        self._struct_bg_idx = (self._struct_bg_idx + 1) % len(_STRUCT_BG_OPTIONS)
        # Apply to the canvas's actual background so the depth-mist
        # fade matches what's behind the structure visually.
        try:
            canvas = self.query_one(
                "#struct-gallery-canvas", _StructureGalleryCanvas,
            )
            _, bg_hex = _STRUCT_BG_OPTIONS[self._struct_bg_idx]
            if bg_hex is None:
                # Fall back to the panel surface — clear any inline override.
                canvas.styles.background = None
            else:
                canvas.styles.background = bg_hex
        except Exception:
            pass
        self._struct_update_title()
        self._struct_refresh_canvas()

    def _struct_tick(self) -> None:
        if not self._struct_rotating or self._struct_coords is None:
            return
        self._struct_angle_y = (
            self._struct_angle_y + self._STRUCT_DEGREES_PER_FRAME
        ) % 360.0
        try:
            self.query_one(
                "#struct-gallery-canvas", _StructureGalleryCanvas,
            ).refresh()
        except Exception:
            pass

    def _struct_load_current(self) -> None:
        """Pull the current CIF's pre-aligned coords from the cache.

        All the heavy lifting (file parsing, sequence alignment,
        Kabsch superposition) happened once in `_struct_build_cache`
        on mount — switching with ←/→ here is just a dict lookup.
        """
        # Local helper: error branches need to wipe *every* per-cif
        # field, otherwise navigating onto a failed cif would leave
        # the title showing the previous structure's pLDDT mean +
        # ligand count (because those state vars retain their last
        # set values).
        def _clear_struct_state(error_msg: str) -> None:
            self._struct_error = error_msg
            self._struct_coords = None
            self._struct_plddt = None
            self._struct_rmsd = None
            self._struct_n_matched = None
            self._struct_ligands = []

        if not self._struct_cifs:
            _clear_struct_state("No structures cached for this project.")
            self._struct_update_title()
            self._struct_refresh_canvas()
            return

        entry = self._struct_cache.get(self._struct_idx)
        if entry is None:
            _clear_struct_state("Structure not in cache.")
        elif entry.get("error"):
            _clear_struct_state(entry["error"])
        else:
            self._struct_coords = entry["coords"]
            self._struct_plddt = entry["plddt"]
            self._struct_rmsd = entry["rmsd"]
            self._struct_n_matched = entry["n_matched"]
            self._struct_ligands = entry.get("ligands") or []
            self._struct_error = None
        self._struct_update_title()
        self._struct_refresh_canvas()

    def _struct_update_title(self) -> None:
        try:
            label = self.query_one("#struct-gallery-title", Label)
            hint = self.query_one("#struct-gallery-hint", Label)
        except Exception:
            return
        if not self._struct_cifs:
            label.update(
                "[dim](no structures cached for this project)[/dim]"
            )
            return
        cif = self._struct_cifs[self._struct_idx]
        n_res = (
            len(self._struct_plddt) if self._struct_plddt is not None else "?"
        )
        mean_plddt = (
            f"  ·  pLDDT {self._struct_plddt.mean():.0f}"
            if self._struct_plddt is not None else ""
        )
        bg_name, _ = _STRUCT_BG_OPTIONS[self._struct_bg_idx]
        # Matchmaker fit quality vs reference (cif #0). The reference
        # has rmsd=0 and shows as "ref"; others show "rmsd 1.23 Å · 87
        # matched", or "no fit" when sequence alignment was too poor.
        if self._struct_idx == 0:
            fit_str = "  ·  [#2E86AB]ref[/#2E86AB]"
        elif self._struct_rmsd is None:
            fit_str = "  ·  [yellow]no fit (PCA fallback)[/yellow]"
        else:
            n_str = (
                f"  ·  {self._struct_n_matched} matched"
                if self._struct_n_matched else ""
            )
            fit_str = (
                f"  ·  rmsd [bold]{self._struct_rmsd:.2f}[/bold] Å"
                f"{n_str}"
            )
        # Ligand count: `[yellow]2 ligands · 47 atoms[/yellow]` when
        # the cif has bound non-polymers worth showing. Empty for AF.
        n_ligs = len(self._struct_ligands)
        n_lig_atoms = sum(len(g["coords"]) for g in self._struct_ligands)
        lig_str = (
            f"  ·  [#FFD93D]{n_ligs} ligand{'s' if n_ligs != 1 else ''} "
            f"· {n_lig_atoms} atoms[/#FFD93D]"
            if n_ligs else ""
        )
        # Pinned-default badge when this cif is the project's
        # preferred structure for the main detail screen.
        prefs = self._project.manifest().get("view") or {}
        is_default = prefs.get("preferred_structure") == cif.name
        default_badge = (
            "  ·  [bold green]★ default[/bold green]"
            if is_default else ""
        )
        label.update(
            f"[bold #2E86AB]{cif.stem}[/bold #2E86AB]  "
            f"[dim]· {self._struct_idx + 1}/{len(self._struct_cifs)} "
            f"· {n_res} res{mean_plddt}{fit_str}{lig_str}{default_badge}"
            f"  ·  view {self._struct_view_mode}  ·  bg {bg_name}[/dim]"
        )
        hint.update(
            "[dim]← →  switch · space  pause · v  view · b  bg "
            "· f  fetch alts · m  make default[/dim]"
        )

    def _struct_refresh_canvas(self) -> None:
        try:
            self.query_one(
                "#struct-gallery-canvas", _StructureGalleryCanvas,
            ).refresh()
        except Exception:
            pass

    def _render_structure_canvas(self, w: int, h: int):
        # Returns a Rich `Text` for the heavy structure render (or a
        # markup string for the cheap status / error placeholders); see
        # the comment in `StructureView._render_canvas` — same rationale,
        # avoiding the markup tokenizer on the hot paint path.
        if self._struct_error:
            return f"[red]{self._struct_error}[/red]"
        if self._struct_coords is None or w < 4 or h < 2:
            return "[dim]Loading…[/dim]"
        from math import radians
        from ..structure import render_structure
        # `default` bg uses the modal panel's actual rendered background
        # (queried from the canvas widget's `styles.background`) so the
        # depth-mist fades toward the surrounding panel rather than a
        # hard-coded gray. The other entries in `_STRUCT_BG_OPTIONS`
        # carry their own explicit hex.
        _, preset_hex = _STRUCT_BG_OPTIONS[self._struct_bg_idx]
        bg_hex = preset_hex
        if bg_hex is None:
            try:
                canvas = self.query_one(
                    "#struct-gallery-canvas", _StructureGalleryCanvas,
                )
                bg = canvas.styles.background
                if bg is not None:
                    bg_hex = bg.hex
            except Exception:
                pass
        return render_structure(
            self._struct_coords, self._struct_plddt, w, h,
            angle_y=radians(self._struct_angle_y),
            angle_x=0.0,
            color_mode="plddt",
            view_mode=self._struct_view_mode,
            bg_color=bg_hex,
            ligand_groups=self._struct_ligands or None,
        )

    # ---- reset helpers ----

    def _reset_alignment_for(self, set_name: str | None) -> None:
        """Clear alignment artefacts of `set_name`; keep hits + search.

        Cascades to the remote: any align_job_id recorded for the set
        gets its `~/beak_jobs/<id>/` rm'd as part of the reset.
        """
        if not set_name:
            set_name = self._project.active_set_name()
        h_dir = self._project.homologs_set_dir(set_name)
        from ...alignments.cache import invalidate_cache
        # Conservation cache filename moved to `_jsd.npy` after the
        # 2026 switch from target-identity to JSD; the legacy plain
        # name is included here too so reset works on projects that
        # haven't been touched since the upgrade.
        for name in ("alignment.fasta", "conservation_jsd.npy", "conservation.npy"):
            f = h_dir / name
            if f.exists():
                if name == "alignment.fasta":
                    invalidate_cache(f)
                f.unlink()

        # Snapshot align_job_id before the mutate clears it.
        align_jids: List[str] = []
        for s in self._project.homologs_sets():
            if s.get("name") == set_name:
                jid = (s.get("remote") or {}).get("align_job_id")
                if jid:
                    align_jids.append(jid)

        with self._project.mutate() as m:
            homologs = m.setdefault("homologs", {})
            sets = homologs.get("sets") or []
            for i, s in enumerate(sets):
                if s.get("name") == set_name:
                    s.pop("n_aligned", None)
                    remote = s.get("remote") or {}
                    remote.pop("align_job_id", None)
                    s["remote"] = remote
                    sets[i] = s
                    break

        if align_jids:
            self._cleanup_remote_jobs(align_jids)
        self.dismiss("reset-align")

    def _reset_homologs(self) -> None:
        """Wipe the active set entirely — hits, alignment, taxonomy cache.

        Cascades to the remote: search/align job dirs for the set get
        nuked, plus any embedding job dirs that were computed against
        this set (since the embeddings entries are also discarded).
        """
        import shutil
        active_name = self._project.active_set_name()
        h_dir = self._project.active_homologs_dir()
        if h_dir.exists():
            shutil.rmtree(h_dir)

        # Collect every job_id we're about to orphan — search, align,
        # and any embedding model computed against this homolog set.
        cleanup_jids: List[str] = []
        for s in self._project.homologs_sets():
            if s.get("name") != active_name:
                continue
            remote = s.get("remote") or {}
            for k in ("search_job_id", "align_job_id"):
                jid = remote.get(k)
                if jid:
                    cleanup_jids.append(jid)
        for e in self._project.embeddings_sets():
            if e.get("source_homologs_set") != active_name:
                continue
            jid = (e.get("remote") or {}).get("job_id")
            if jid:
                cleanup_jids.append(jid)

        with self._project.mutate() as m:
            homologs = m.get("homologs") or {}
            sets = homologs.get("sets") or []
            sets = [s for s in sets if s.get("name") != active_name]
            if sets:
                homologs["sets"] = sets
                # Make the next remaining set active.
                homologs["active"] = sets[0].get("name", "default")
                m["homologs"] = homologs
            else:
                m.pop("homologs", None)

            # Drop embeddings tied to the dead set too — keeping them
            # would leave dangling source_homologs_set references.
            emb = m.get("embeddings") or {}
            emb_sets = emb.get("sets") or []
            emb_sets = [
                e for e in emb_sets
                if e.get("source_homologs_set") != active_name
            ]
            if emb_sets:
                emb["sets"] = emb_sets
                m["embeddings"] = emb
            elif "embeddings" in m:
                m.pop("embeddings", None)

        if cleanup_jids:
            self._cleanup_remote_jobs(cleanup_jids)
        self.dismiss("reset-homologs")

    # ---- length histogram ----

    # Vertical block ramp — each cell carries 1/8 of a row's worth of
    # bar height, so a `height`-row chart resolves `height * 8` discrete
    # bin levels.
    _HIST_BLOCKS = " ▁▂▃▄▅▆▇█"
    # Bins are sized to the chart pane on first paint; this is the
    # fallback when the pane hasn't been measured yet.
    _HIST_DEFAULT_BINS = 36
    _HIST_DEFAULT_HEIGHT = 8

    def _refresh_length_panel(self, set_name: str) -> None:
        """Render length + %ID histograms for `set_name` into the pane.

        Two stacked charts:
        1. Sequence length distribution — always present whenever
           sequences.fasta is on disk.
        2. %ID-to-target distribution — only when alignment.fasta is
           on disk; computed lazily by `_compute_identities_worker`.

        Both share the same column-block renderer with a `▲` median
        marker below the X-axis so the central tendency reads at a
        glance without having to compare the stats line to the chart.
        """
        fasta = self._project.homologs_set_dir(set_name) / "sequences.fasta"
        if not fasta.exists():
            self._update_length_panel("[dim]No sequences yet.[/dim]")
            return

        if set_name not in self._length_cache:
            if set_name not in self._length_pending:
                self._length_pending.add(set_name)
                self._compute_lengths_worker(set_name)
            self._update_length_panel("[dim]Computing length distribution…[/dim]")
            return

        lengths = self._length_cache[set_name]
        if not lengths:
            self._update_length_panel("[dim]No sequences in FASTA.[/dim]")
            return

        # Two stacked panes: split available height between them so
        # both fit. Length always renders; identity renders below it
        # whenever the alignment is on disk.
        aln_path = self._project.homologs_set_dir(set_name) / "alignment.fasta"
        has_alignment = aln_path.exists()

        bins, height = self._chart_dims(n_charts=2 if has_alignment else 1)
        if has_alignment:
            # Two charts share the height — round up for length so it
            # gets the extra row when the budget is odd.
            len_height = max(3, (height + 1) // 2)
            id_height = max(3, height - len_height)
        else:
            len_height = height
            id_height = 0

        len_chart = self._render_histogram(
            lengths, bins=bins, height=len_height, median_marker=True,
        )
        len_header = (
            f"[bold]Length distribution[/bold]  "
            f"[dim](n={len(lengths):,})[/dim]"
        )
        len_stats = self._length_stats_line(lengths)
        sections = [f"{len_header}\n{len_stats}\n\n{len_chart}"]

        if has_alignment:
            if set_name not in self._identity_cache:
                if set_name not in self._identity_pending:
                    self._identity_pending.add(set_name)
                    self._compute_identities_worker(set_name)
                sections.append(
                    "[bold]%ID to target[/bold]\n"
                    "[dim]Computing identity distribution…[/dim]"
                )
            else:
                ids = self._identity_cache[set_name]
                if not ids:
                    sections.append(
                        "[bold]%ID to target[/bold]\n"
                        "[dim]Alignment too small for identities.[/dim]"
                    )
                else:
                    id_chart = self._render_histogram(
                        ids, bins=bins, height=id_height,
                        median_marker=True,
                        x_unit="%", x_min=0.0, x_max=100.0,
                    )
                    id_header = (
                        f"[bold]%ID to target[/bold]  "
                        f"[dim](n={len(ids):,})[/dim]"
                    )
                    id_stats = self._identity_stats_line(ids)
                    sections.append(
                        f"{id_header}\n{id_stats}\n\n{id_chart}"
                    )
        self._update_length_panel("\n\n".join(sections))

    def _update_length_panel(self, content: str) -> None:
        try:
            panel = self.query_one("#length-hist-panel", Static)
        except Exception:
            return
        panel.update(content)

    def _chart_dims(self, n_charts: int = 1) -> tuple[int, int]:
        """Return (n_bins, total chart height) sized to the histogram pane.

        Each chart section consumes a fixed non-chart overhead:
        1 header + 1 stats + 1 blank gutter + 1 median marker row +
        1 axis-labels row = 5 rows. Stacking two charts adds one more
        blank row between sections (the `\n\n` separator joining the
        section strings). The constant 4 covers the panel's border
        (top + bottom) and vertical padding (1 each side).

        Without `n_charts`, the previous version would return a budget
        sized for one chart even when two were rendered, so the second
        one fell off the bottom of the pane (visible as a 1-row stub
        in the homologs modal).
        """
        try:
            panel = self.query_one("#length-hist-panel", Static)
            w = max(20, panel.size.width - 4)  # 2 padding on each side
            chrome = 4 + 5 * n_charts + max(0, n_charts - 1)
            h = max(n_charts * 3, panel.size.height - chrome)
        except Exception:
            return self._HIST_DEFAULT_BINS, self._HIST_DEFAULT_HEIGHT
        return min(60, w), min(16 * n_charts, h)

    @staticmethod
    def _length_stats_line(lengths: list[int]) -> str:
        sorted_l = sorted(lengths)
        lo, hi = sorted_l[0], sorted_l[-1]
        median = sorted_l[len(sorted_l) // 2]
        mean = sum(sorted_l) / len(sorted_l)
        return (
            f"[dim]min[/dim] {lo}  "
            f"[dim]median[/dim] {median}  "
            f"[dim]mean[/dim] {mean:.0f}  "
            f"[dim]max[/dim] {hi}  [dim]aa[/dim]"
        )

    def _render_histogram(
        self,
        values,
        bins: int,
        height: int,
        median_marker: bool = False,
        x_unit: str = "aa",
        x_min: float = None,
        x_max: float = None,
    ) -> str:
        """Block-ramp histogram with optional `▲` median marker.

        `x_min` / `x_max` pin the axis range explicitly (used for the
        %ID chart, where the natural [0, 100] range gives a fairer
        comparison across sets even when one set's lo/hi happen to
        be tighter).
        """
        if not values:
            return "[dim]no data[/dim]"
        lo = float(x_min) if x_min is not None else float(min(values))
        hi = float(x_max) if x_max is not None else float(max(values))
        if hi <= lo:
            return f"[dim]all {lo:g} {x_unit}[/dim]"
        span = hi - lo
        counts = [0] * bins
        for v in values:
            idx = int((float(v) - lo) / span * bins)
            if idx >= bins:
                idx = bins - 1
            elif idx < 0:
                idx = 0
            counts[idx] += 1
        peak = max(counts) or 1

        total_units = height * 8
        rows: list[str] = []
        for r in range(height - 1, -1, -1):
            cells = []
            for c in counts:
                col_units = c / peak * total_units
                cell_units = int(round(col_units - r * 8))
                cell_units = max(0, min(8, cell_units))
                cells.append(self._HIST_BLOCKS[cell_units])
            rows.append(f"[#65CBF3]{''.join(cells)}[/#65CBF3]")

        # Median marker (▲) directly below the bar at the column the
        # median falls into. Drawn on its own row so it can use a
        # contrasting color without disturbing the gradient stack.
        if median_marker:
            sorted_v = sorted(values)
            median = float(sorted_v[len(sorted_v) // 2])
            m_idx = int((median - lo) / span * bins)
            m_idx = max(0, min(bins - 1, m_idx))
            cells = [" "] * bins
            cells[m_idx] = "▲"
            rows.append(f"[#FFA62B]{''.join(cells)}[/#FFA62B]")

        # X-axis: lo on left, hi on right. Use formatted width so the
        # %ID chart shows "0%" / "100%" cleanly.
        left = f"{lo:g}"
        right = f"{hi:g} {x_unit}".strip()
        gap = max(1, bins - len(left) - len(right))
        rows.append(f"[dim]{left}{' ' * gap}{right}[/dim]")
        return "\n".join(rows)

    @work(thread=True, group="length-hist")
    def _compute_lengths_worker(self, set_name: str) -> None:
        """Parse hit FASTA into a list of sequence lengths.

        Streams line-by-line so a 40 MB hits file doesn't pull into
        memory all at once; only the integer lengths are kept.
        """
        fasta = self._project.homologs_set_dir(set_name) / "sequences.fasta"
        lengths: list[int] = []
        try:
            cur = 0
            with open(fasta) as f:
                for line in f:
                    if line.startswith(">"):
                        if cur:
                            lengths.append(cur)
                        cur = 0
                    else:
                        cur += len(line.strip())
                if cur:
                    lengths.append(cur)
        except Exception:
            lengths = []

        self._length_cache[set_name] = lengths
        self._length_pending.discard(set_name)
        # Refresh only if the user is still on this row — otherwise the
        # cached entry will be picked up the next time they tab back.
        try:
            self.app.call_from_thread(self._refresh_if_selected, set_name)
        except Exception:
            pass

    @staticmethod
    def _identity_stats_line(values: list) -> str:
        """One-liner stats for the %ID histogram pane."""
        if not values:
            return ""
        sorted_v = sorted(values)
        lo, hi = sorted_v[0], sorted_v[-1]
        median = sorted_v[len(sorted_v) // 2]
        mean = sum(sorted_v) / len(sorted_v)
        return (
            f"[dim]min[/dim] {lo:.0f}%  "
            f"[dim]median[/dim] [#FFA62B]{median:.0f}%[/#FFA62B]  "
            f"[dim]mean[/dim] {mean:.0f}%  "
            f"[dim]max[/dim] {hi:.0f}%"
        )

    @work(thread=True, group="identity-hist")
    def _compute_identities_worker(self, set_name: str) -> None:
        """Compute %ID-to-target across the alignment for `set_name`.

        Each homolog row's identity is the fraction of non-gap target
        columns at which the homolog matches the target letter.
        Skips the target row itself and rows whose alignment length
        differs from the target's (malformed records).
        """
        aln_path = self._project.homologs_set_dir(set_name) / "alignment.fasta"
        ids: list[float] = []
        try:
            from ...alignments.cache import load_alignment_records
            # Same .npz cache used by `_render_set_details` —
            # parsing the alignment via BioPython directly was the
            # second-largest contributor to slow modal opens (the
            # first being the synchronous render path that's now
            # also cached). On a 7k-seq alignment this drops the
            # worker's wall-clock from ~3s to ~80ms cold, free hot.
            records = [
                (name, seq) for name, seq in load_alignment_records(aln_path)
            ]
            if len(records) >= 2:
                target_seq = records[0][1].upper()
                n = len(target_seq)
                target_mask = [
                    c not in "-." for c in target_seq
                ]
                n_real = sum(target_mask)
                if n_real:
                    target_chars = list(target_seq)
                    for _name, raw_seq in records[1:]:
                        seq = raw_seq.upper()
                        if len(seq) != n:
                            continue
                        matches = 0
                        for j, m in enumerate(target_mask):
                            if m and seq[j] == target_chars[j]:
                                matches += 1
                        ids.append(matches / n_real * 100.0)
        except Exception:
            ids = []

        self._identity_cache[set_name] = ids
        self._identity_pending.discard(set_name)
        try:
            self.app.call_from_thread(self._refresh_if_selected, set_name)
        except Exception:
            pass

    def _refresh_if_selected(self, set_name: str) -> None:
        if self._selected_set == set_name:
            # Only the histogram pane needs an update — the text details
            # don't carry the histogram anymore, so this avoids a
            # redundant re-render of the file/taxonomy block.
            self._refresh_length_panel(set_name)

    def _disarm_remove_embed(self) -> None:
        """Reset the Remove button label after a context switch."""
        self._remove_embed_armed = False
        try:
            btn = self.query_one("#remove-embedding-btn", Button)
            btn.label = "Remove"
        except Exception:
            pass

    def _remove_embedding_clicked(self, button: Button) -> None:
        """Two-click confirm for the row-targeted Remove.

        First click arms the button (label flips to 'Confirm remove')
        and validates that a row is selected. Second click reads the
        selected row's (set, model) entry, kicks off remote cleanup
        for *just that row's* job_id, drops the manifest entry +
        per-(set, model) directory, then dismisses so the parent
        layers panel refreshes.
        """
        entry = self._selected_embed_entry()
        if entry is None:
            self.notify(
                "Select a row to remove. Use ↑/↓ to highlight one.",
                severity="warning", timeout=4,
            )
            return

        if not self._remove_embed_armed:
            set_name = entry.get("source_homologs_set") or "?"
            model = (entry.get("model") or "-").split("/")[-1]
            self._remove_embed_armed = True
            button.label = "Confirm remove"
            self.notify(
                f"Click again to remove · set={set_name} · model={model}",
                timeout=5,
            )
            return

        # Second click: do it.
        self._remove_embed_armed = False
        set_name = entry.get("source_homologs_set")
        model = entry.get("model")
        job_id = (entry.get("remote") or {}).get("job_id")

        if not set_name:
            self.notify(
                "Selected row has no source_homologs_set — manifest "
                "is malformed; close and reopen.",
                severity="error", timeout=8,
            )
            return

        # Kick off remote cleanup *before* the local delete so the
        # worker is registered while the modal is still alive.
        if job_id:
            self._cleanup_remote_jobs([job_id])

        self._project.delete_embeddings_set(set_name, model=model)
        self.dismiss("reset-embeddings")

    def _reset_embeddings(self) -> None:
        """Wipe the active set's embeddings — local files + manifest +
        remote job dirs.

        Remote cleanup runs in a background worker so the modal closes
        immediately. Best-effort: if SSH is unreachable, the local delete
        still completes and the user sees a warning toast pointing at
        the remote dirs they'd need to remove by hand.
        """
        # Snapshot the job ids about to be removed *before* the local
        # delete clears them from the manifest.
        active = self._project.active_set_name()
        job_ids = [
            (e.get("remote") or {}).get("job_id")
            for e in self._project.embeddings_sets()
            if e.get("source_homologs_set") == active
        ]
        job_ids = [j for j in job_ids if j]

        # Kick off remote cleanup before the local delete so the worker
        # is registered while this modal is still alive — Textual
        # workers belong to the app, not the modal, so it'll continue
        # running after dismiss.
        if job_ids:
            self._cleanup_remote_jobs(job_ids)

        self._project.delete_active_embeddings_set()
        self.dismiss("reset-embeddings")

    @work(thread=True, exclusive=True, group="cleanup-jobs")
    def _cleanup_remote_jobs(self, job_ids: List[str]) -> None:
        """rm -rf each job's remote dir and remove its jobs.json entry.

        Works for any job kind (search / align / embed / tax) — the
        `cleanup()` method on `RemoteJobManager` only relies on the
        per-job `remote_path` recorded in jobs.json, and that path is
        validated against the manager's `remote_job_dir` server-side.
        Picking ESMEmbeddings as the carrier is convenient (it
        verifies docker on init, but the cleanup itself is generic).
        """
        if not job_ids:
            return
        from ...remote.embeddings import ESMEmbeddings

        mgr = None
        failed: List[str] = []
        try:
            try:
                mgr = ESMEmbeddings()
            except Exception as e:  # noqa: BLE001
                self.app.call_from_thread(
                    self.notify,
                    f"Remote cleanup skipped — can't reach the server "
                    f"({type(e).__name__}). {len(job_ids)} job dir(s) "
                    f"left on remote.",
                    severity="warning",
                    timeout=12,
                )
                return

            for jid in job_ids:
                try:
                    mgr.cleanup(jid, keep_results=False)
                except Exception:  # noqa: BLE001
                    failed.append(jid)

            if failed:
                self.app.call_from_thread(
                    self.notify,
                    f"Remote cleanup partial: {len(failed)}/{len(job_ids)} "
                    f"job dir(s) left behind ({', '.join(failed[:3])}"
                    f"{'…' if len(failed) > 3 else ''}).",
                    severity="warning",
                    timeout=10,
                )
            elif job_ids:
                self.app.call_from_thread(
                    self.notify,
                    f"Cleaned up {len(job_ids)} remote job dir"
                    f"{'s' if len(job_ids) > 1 else ''}.",
                    timeout=4,
                )
        finally:
            if mgr is not None:
                try:
                    conn = getattr(mgr, "conn", None)
                    if conn is not None:
                        conn.close()
                except Exception:  # noqa: BLE001
                    pass
