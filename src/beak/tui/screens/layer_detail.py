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


class LayerDetailModal(ModalScreen):
    """Click a layer row -> this opens with the layer's details."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, project: BeakProject, layer_name: str) -> None:
        super().__init__()
        self._project = project
        self._layer = layer_name
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
        # Cache of per-set sequence lengths for the length histogram —
        # populated by `_compute_lengths_worker` so tabbing between sets
        # only pays the parse cost on first view of each.
        self._length_cache: dict[str, list[int]] = {}
        self._length_pending: set[str] = set()
        # Set when an in-modal action mutates a set (filter, rename, …)
        # without dismissing the modal. On close we surface a marker to
        # the parent so the layers panel + views refresh.
        self._dirty: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(f"[bold]Layer · {self._layer}[/bold]", id="layer-detail-title")
            if self._layer == "homologs":
                yield from self._compose_homologs()
            elif self._layer == "embeddings":
                yield from self._compose_embeddings()
            else:
                yield Static(self._render_text(), id="layer-detail-content")
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
        with Horizontal(id="modal-buttons"):
            with Horizontal(classes="btn-group"):
                yield Button(
                    "Activate", id="make-active-btn",
                    variant="primary", compact=True,
                    tooltip=(
                        "Use this set's hits and alignment in the project views."
                    ),
                )
            with Horizontal(classes="btn-group"):
                yield Button(
                    "Rename", id="rename-set-btn", compact=True,
                    tooltip="Rename this set.",
                )
                yield Button(
                    "Filter", id="filter-length-btn", compact=True,
                    tooltip=(
                        "Trim hits by sequence length — clears the alignment "
                        "for this set so you can re-align the trimmed FASTA."
                    ),
                )
                yield Button(
                    "Dedupe", id="dedupe-set-btn", compact=True,
                    tooltip=(
                        "Drop exact-duplicate sequences (case-insensitive). "
                        "Clears the existing alignment if anything is removed."
                    ),
                )
            with Horizontal(classes="btn-group"):
                yield Button(
                    "Align", id="submit-align-btn", compact=True,
                    tooltip=(
                        "Submit a Clustal Omega alignment for this set's hits. "
                        "Replaces any existing alignment."
                    ),
                )
                yield Button(
                    "Taxonomy", id="build-tax-btn", compact=True,
                    tooltip="Re-fetch UniProt taxonomy for this set's hits.",
                )
                yield Button(
                    "Drop align", id="reset-align-btn", compact=True,
                    tooltip=(
                        "Delete the alignment but keep the hits. "
                        "Re-aligning replaces it."
                    ),
                )
            with Horizontal(classes="btn-group"):
                yield Button(
                    "Delete set", id="delete-set-btn",
                    variant="warning", compact=True,
                    tooltip=(
                        "Permanently delete this set's hits, alignment, "
                        "and metadata."
                    ),
                )
            # Pushes the modal-level actions to the right edge.
            yield Static("", id="btn-spacer-grow")
            with Horizontal(classes="btn-group"):
                yield Button(
                    "Search", id="new-search-btn", compact=True,
                    tooltip="Run a new database search — creates another set.",
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

    def _render_set_details(self, set_name: str) -> str:
        """Per-set details panel: runs when the table cursor moves."""
        set_dict = next(
            (s for s in self._project.homologs_sets()
             if s.get("name") == set_name),
            None,
        )
        if set_dict is None:
            return ""
        remote = set_dict.get("remote") or {}
        rows = []
        rows.append(self._kv("Source", set_dict.get("source", "-")))
        rows.append(self._kv("Database", remote.get("search_database", "-")))
        n_homologs = set_dict.get("n_homologs", 0)
        rows.append(self._kv("Hits", f"{n_homologs:,}" if n_homologs else "-"))
        n_aligned = set_dict.get("n_aligned", 0)
        if n_aligned:
            aligned_str = f"{n_aligned:,}"
        elif remote.get("align_job_id"):
            aligned_str = "running"
        else:
            aligned_str = "-"
        rows.append(self._kv("Aligned", aligned_str))

        # Diversity metrics — only meaningful once alignment has landed.
        aln_path = self._project.homologs_set_dir(set_name) / "alignment.fasta"
        if aln_path.exists():
            try:
                from .alignment_view import _mean_identity_to_target, _effective_n
                from Bio import SeqIO
                seqs = [(r.id, str(r.seq)) for r in SeqIO.parse(str(aln_path), "fasta")]
                ident = _mean_identity_to_target(seqs)
                neff = _effective_n(seqs, threshold=0.8)
                if ident is not None:
                    rows.append(self._kv("Mean identity", f"{ident:.0f}%"))
                if neff is not None:
                    rows.append(self._kv("Neff @80%", f"{neff} clusters"))
                if ident is not None and ident > 90:
                    rows.append(self._kv("Note", "very similar sequences (low diversity)"))
                elif neff is not None and n_aligned and neff < n_aligned * 0.1:
                    rows.append(self._kv("Note", "high redundancy"))
            except Exception:
                pass

        # Taxonomy summary — top phyla + temperature range.
        tax_path = self._project.homologs_set_dir(set_name) / "taxonomy.parquet"
        if tax_path.exists():
            try:
                import pandas as pd
                tax = pd.read_parquet(tax_path)
                rows.append("")
                rows.append("  Taxonomy:")
                if "domain" in tax.columns:
                    domain_counts = tax["domain"].dropna().value_counts().head(3)
                    if len(domain_counts):
                        rows.append("    Domains: " + ", ".join(
                            f"{d} ({n})" for d, n in domain_counts.items()
                        ))
                if "phylum" in tax.columns:
                    phylum_counts = tax["phylum"].dropna().value_counts().head(5)
                    if len(phylum_counts):
                        rows.append("    Top phyla: " + ", ".join(
                            f"{p} ({n})" for p, n in phylum_counts.items()
                        ))
                if "growth_temp" in tax.columns:
                    temps = tax["growth_temp"].dropna()
                    if len(temps):
                        rows.append(
                            f"    Growth temp: {temps.min():.0f}-{temps.max():.0f} C "
                            f"(n={len(temps)})"
                        )
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
                    rows.append("")
                    rows.append("  Traits (metaTraits):")
                    if "trait_match_level" in tr.columns:
                        n_matched = int(tr["trait_match_level"].notna().sum())
                        levels = tr["trait_match_level"].dropna().value_counts()
                        breakdown = ", ".join(
                            f"{lvl}: {n}" for lvl, n in levels.items()
                        )
                        rows.append(
                            f"    Matched: {n_matched}/{len(tr)}"
                            + (f"  ({breakdown})" if breakdown else "")
                        )
                    # Sample up to 5 informative columns: prefer ones with
                    # multiple non-null values to avoid showing all-blank rows.
                    coverage = [
                        (c, int(tr[c].notna().sum())) for c in trait_cols
                    ]
                    coverage.sort(key=lambda kv: kv[1], reverse=True)
                    shown = 0
                    for col, cov in coverage:
                        if cov == 0 or shown >= 5:
                            continue
                        s = tr[col].dropna()
                        nice = col.removeprefix("trait_").replace("_", " ")
                        if pd.api.types.is_numeric_dtype(s):
                            rows.append(
                                f"    {nice}: {s.min():.2g}-{s.max():.2g} "
                                f"(n={cov})"
                            )
                        else:
                            top = s.astype(str).value_counts().head(3)
                            rows.append(
                                f"    {nice}: " + ", ".join(
                                    f"{v} ({n})" for v, n in top.items()
                                )
                            )
                        shown += 1
            except Exception:
                pass

        last = set_dict.get("last_updated")
        if last is not None:
            rows.append(self._kv("Last updated", _format_dt(last)))

        s_jid = remote.get("search_job_id")
        a_jid = remote.get("align_job_id")
        if s_jid or a_jid:
            rows.append("")
            if s_jid:
                rows.append(self._kv("Search job", f"{s_jid}  -  {_job_status(s_jid)}"))
            if a_jid:
                rows.append(self._kv("Align job", f"{a_jid}  -  {_job_status(a_jid)}"))

        homologs_dir = self._project.homologs_set_dir(set_name)
        if homologs_dir.exists():
            files = sorted(p for p in homologs_dir.iterdir() if p.is_file())
            if files:
                rows.append("")
                rows.append("  Files:")
                for p in files:
                    rows.append(f"    - {p.name}  ({_human_size(p.stat().st_size)})")

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

        yield Label("[bold]Embeddings[/bold]", classes="section-label")
        table = DataTable(
            id="embeddings-table", cursor_type="row", zebra_stripes=True,
        )
        table.add_columns(
            " ", "Set", "Model", "Files", "Last updated", "Status",
        )
        for i, e in enumerate(entries):
            src = e.get("source_homologs_set") or "?"
            marker = "★" if src == active_name else ""
            # Strip HF org prefixes for compactness.
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
            table.add_row(marker, src, model, n_str, last, status, key=row_key)
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
            self.dismiss("set-mutated" if self._dirty else None)
        elif bid == "reset-align-btn":
            self._reset_alignment_for(self._selected_set)
        elif bid == "reset-homologs-btn":
            self._reset_homologs()
        elif bid == "remove-embedding-btn":
            self._remove_embedding_clicked(event.button)
        elif bid == "build-tax-btn":
            self._build_taxonomy_now(event.button)
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
            # parent refreshes layers/views on close.
            self._length_cache.pop(set_name, None)
            self._length_pending.discard(set_name)
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
            # Invalidate cached lengths so the histogram re-parses the
            # trimmed FASTA on the next refresh.
            self._length_cache.pop(set_name, None)
            self._length_pending.discard(set_name)
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

        try:
            tax_btn = self.query_one("#build-tax-btn", Button)
            tax_path = (
                self._project.homologs_set_dir(self._selected_set)
                / "taxonomy.parquet"
            )
            has_hits = bool(set_dict.get("n_homologs"))
            tax_btn.disabled = not has_hits or tax_path.exists()
        except Exception:
            pass

        try:
            align_btn = self.query_one("#reset-align-btn", Button)
            align_btn.disabled = not bool(set_dict.get("n_aligned"))
        except Exception:
            pass

        # The "Align" button (re-align too) — enabled when:
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
        self.dismiss("set-mutated" if self._dirty else None)

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
        for name in ("alignment.fasta", "conservation.npy"):
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
        """Push the length-histogram chart for `set_name` into its pane."""
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

        # Scale bins/height to the available chart pane so the bars
        # actually use the corner the user just dedicated to them.
        bins, height = self._chart_dims()
        chart = self._render_histogram(lengths, bins=bins, height=height)
        header = (
            f"[bold]Length distribution[/bold]  "
            f"[dim](n={len(lengths):,})[/dim]"
        )
        stats = self._length_stats_line(lengths)
        self._update_length_panel(f"{header}\n{stats}\n\n{chart}")

    def _update_length_panel(self, content: str) -> None:
        try:
            panel = self.query_one("#length-hist-panel", Static)
        except Exception:
            return
        panel.update(content)

    def _chart_dims(self) -> tuple[int, int]:
        """Return (n_bins, height) sized to the histogram pane."""
        try:
            panel = self.query_one("#length-hist-panel", Static)
            w = max(20, panel.size.width - 4)  # 2 padding on each side
            h = max(4, panel.size.height - 6)  # header + stats + axis labels
        except Exception:
            return self._HIST_DEFAULT_BINS, self._HIST_DEFAULT_HEIGHT
        return min(60, w), min(16, h)

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
        self, lengths: list[int], bins: int, height: int
    ) -> str:
        # Defensive: callers gate on `lengths` truthiness, but if they
        # ever slip through with [] we'd crash on `min([])` / `max([])`.
        if not lengths:
            return "[dim]no length data[/dim]"
        lo = min(lengths)
        hi = max(lengths)
        if hi == lo:
            return f"[dim]all {lo} aa[/dim]"
        span = hi - lo
        counts = [0] * bins
        for L in lengths:
            idx = int((L - lo) / span * bins)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        peak = max(counts) or 1

        # For each row from the top down, decide how full each column is.
        # `height * 8` total sub-cells per column — block ramp resolves
        # 8 levels per cell, giving a smooth bar even at small `height`.
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

        # X-axis: min on the left, max on the right, padded to bar width.
        left = str(lo)
        right = f"{hi} aa"
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
