"""Modal showing detailed info for a project layer."""

import json
from datetime import datetime
from pathlib import Path

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

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(f"[bold]Layer · {self._layer}[/bold]", id="layer-detail-title")
            if self._layer == "homologs":
                yield from self._compose_homologs()
            else:
                yield Static(self._render_text(), id="layer-detail-content")
                with Horizontal(id="modal-buttons"):
                    m = self._project.manifest()
                    if self._layer == "embeddings":
                        if (m.get("embeddings") or {}).get("n_embeddings"):
                            yield Button("Reset", id="reset-embeddings-btn",
                                         variant="warning")
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

        # Per-row details panel lives in a scroller so the modal's overall
        # height doesn't shift when the highlighted set's content gets
        # longer/shorter (taxonomy / files / job blocks vary). The
        # scroller takes the remaining vertical space inside #modal-body.
        with VerticalScroll(id="set-details-scroll"):
            yield Static("", id="set-details", classes="set-details")

        # Action buttons. Compact so all seven fit a 120-cell modal
        # without overflowing into the right border.
        with Horizontal(id="modal-buttons"):
            yield Button("New search", id="new-search-btn", compact=True)
            yield Button(
                "Make active", id="make-active-btn",
                variant="primary", compact=True,
            )
            yield Button("Rename", id="rename-set-btn", compact=True)
            yield Button("Build taxonomy", id="build-tax-btn", compact=True)
            yield Button("Reset align", id="reset-align-btn", compact=True)
            yield Button(
                "Delete set", id="delete-set-btn",
                variant="warning", compact=True,
            )
            yield Button("Close", id="close-btn", compact=True)

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

    def _render_embeddings(self, m: dict) -> str:
        e = m.get("embeddings") or {}
        remote = e.get("remote") or {}
        rows = []
        rows.append(self._kv("Model", e.get("model", "-") or "-"))
        n = e.get("n_embeddings", 0)
        rows.append(self._kv("Files", f"{n:,}" if n else "-"))
        last = e.get("last_updated")
        if last is not None:
            rows.append(self._kv("Last updated", _format_dt(last)))
        jid = remote.get("job_id")
        if jid:
            rows.append("")
            rows.append(self._kv("Job", f"{jid}  -  {_job_status(jid)}"))

        emb_dir = self._project.path / "embeddings"
        if emb_dir.exists():
            files = sorted(p for p in emb_dir.rglob("*") if p.is_file())[:8]
            if files:
                rows.append("")
                rows.append("  Sample files:")
                for p in files:
                    rows.append(f"    - {p.relative_to(emb_dir)}  ({_human_size(p.stat().st_size)})")
        return "\n".join(rows) if rows else "No embeddings yet."

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
        if bid == "close-btn":
            self.dismiss(None)
        elif bid == "reset-align-btn":
            self._reset_alignment_for(self._selected_set)
        elif bid == "reset-homologs-btn":
            self._reset_homologs()
        elif bid == "reset-embeddings-btn":
            self._reset_embeddings()
        elif bid == "build-tax-btn":
            self._build_taxonomy_now(event.button)
        elif bid == "make-active-btn":
            self._make_selected_active()
        elif bid == "delete-set-btn":
            self._delete_set_clicked(event.button)
        elif bid == "new-search-btn":
            self._open_new_search()
        elif bid == "rename-set-btn":
            self._open_rename_set()

    def _open_new_search(self) -> None:
        """Dismiss with a marker so the detail screen opens the search modal."""
        self.dismiss("open-search")

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
            return
        details.update(self._render_set_details(self._selected_set))

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
        try:
            from ..taxonomy import build_taxonomy_table
            df = build_taxonomy_table(self._project, force=True)
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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "active-set-select":
            new_name = str(event.value)
            if new_name and new_name != self._project.active_set_name():
                self._project.set_active_set(new_name)
                self.dismiss("set-switched")

    def action_close(self) -> None:
        self.dismiss(None)

    # ---- reset helpers ----

    def _reset_alignment_for(self, set_name: str | None) -> None:
        """Clear alignment artefacts of `set_name`; keep hits + search."""
        if not set_name:
            set_name = self._project.active_set_name()
        h_dir = self._project.homologs_set_dir(set_name)
        for name in ("alignment.fasta", "conservation.npy"):
            f = h_dir / name
            if f.exists():
                f.unlink()

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
        self.dismiss("reset-align")

    def _reset_homologs(self) -> None:
        """Wipe the active set entirely — hits, alignment, taxonomy cache."""
        import shutil
        active_name = self._project.active_set_name()
        h_dir = self._project.active_homologs_dir()
        if h_dir.exists():
            shutil.rmtree(h_dir)

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
        self.dismiss("reset-homologs")

    def _reset_embeddings(self) -> None:
        """Wipe the embeddings layer."""
        import shutil
        e_dir = self._project.path / "embeddings"
        if e_dir.exists():
            shutil.rmtree(e_dir)
        with self._project.mutate() as m:
            m.pop("embeddings", None)
        self.dismiss("reset-embeddings")
