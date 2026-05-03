"""Project list — landing screen of the TUI.

Two-pane layout: a DataTable of projects on the left, a summary panel on
the right that shows the highlighted project's static structure preview
plus a brief data summary (target metadata + layer counts).
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Static

from ...project import BeakProject
from ..widgets.server_status import ServerStatusBar


# Status colors pulled from beak's existing palette (`tui/app.py`,
# `tui/structure.py`) so the project list reads as part of the same
# visual system as the panel borders, structure ribbon, and pill
# affordances.
#   #65CBF3 — beak cyan (also the high-pLDDT band)
#   #FFA62B — amber (also the Pfam element + the search/embed pills)
#   #7DD87D — palette green (one of the DOMAIN_PALETTE chips)
#   #FF6B6B — palette red (helix accent in the SS legend)
_BEAK_CYAN = "#65CBF3"
_BEAK_AMBER = "#FFA62B"
_BEAK_GREEN = "#7DD87D"
_BEAK_RED = "#FF6B6B"

_STATUS_COLORS = {
    "ready":     _BEAK_GREEN,
    "completed": _BEAK_GREEN,
    # Compute / IO-bound stages: cyan. Long-running compute (embed,
    # multi-job): amber. The two-color split mirrors the in-flight
    # pill colors in the layers panel so a glance from the project
    # list to the layer panel reads as the same status.
    "search":    _BEAK_CYAN,
    "align":     _BEAK_CYAN,
    "tax":       _BEAK_CYAN,
    "submitted": _BEAK_CYAN,
    "queued":    _BEAK_CYAN,
    "embed":     _BEAK_AMBER,
    "running":   _BEAK_AMBER,
    "failed":    _BEAK_RED,
    "cancelled": "dim",
    "new":       "dim",
}


def _human_size(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def _format_dt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)


def _format_status(status: str) -> str:
    color = _STATUS_COLORS.get(status, "dim")
    return f"[{color}]{status}[/{color}]"


def _relative_age(ts: float) -> str:
    """Compact relative timestamp ("2h", "3d", "5w"). 0 → '—'."""
    if not ts:
        return "—"
    import time
    secs = max(0.0, time.time() - ts)
    if secs < 60:
        return "just now"
    mins = secs / 60
    if mins < 60:
        return f"{int(mins)}m"
    hours = mins / 60
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24
    if days < 14:
        return f"{int(days)}d"
    weeks = days / 7
    if weeks < 9:
        return f"{int(weeks)}w"
    months = days / 30
    if months < 18:
        return f"{int(months)}mo"
    return f"{int(days / 365)}y"


def _shorten(text: str, width: int) -> str:
    """Trim a string to `width` cells with an ellipsis if it overflows."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


class ProjectListScreen(Screen):
    BINDINGS = [
        Binding("n", "new_project", "New"),
        Binding("e", "rename_project", "Rename"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "settings", "Settings"),
    ]

    DEFAULT_CSS = """
    ProjectListScreen #list-body {
        height: 1fr;
        width: 100%;
    }
    ProjectListScreen #projects-table {
        width: 55%;
        height: 1fr;
    }
    ProjectListScreen #project-summary {
        width: 45%;
        height: 1fr;
        padding: 1 2;
        border: round #2E86AB;
        margin: 0 1 0 0;
    }
    ProjectListScreen #summary-meta {
        height: auto;
        padding-bottom: 1;
    }
    ProjectListScreen #summary-ribbon {
        height: 1fr;
        min-height: 8;
    }
    ProjectListScreen #summary-stats {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._projects: List[BeakProject] = []
        # Mtime-keyed cache: { project_name: (manifest_mtime, row_dict) }.
        # `_populate` only re-reads a project's manifest when its TOML
        # mtime has changed since the last refresh — for users with 50+
        # projects this turns the typical refresh from 50 file reads
        # into 0.
        self._row_cache: dict = {}
        # Marquee state for the Detail column. Each project carries
        # one long ticker string (built from its in-flight jobs +
        # settled summary, joined by separators) and a per-row
        # character offset that advances on every tick. The cell
        # content is a window onto the ticker; when offset wraps past
        # the end, a doubled string keeps the window seamless.
        self._detail_text: Dict[str, str] = {}
        self._detail_color: Dict[str, str] = {}
        self._detail_offset: Dict[str, int] = {}
        # Ticks remaining of "stay at offset 0" dwell. Set on init and
        # whenever the offset wraps back to 0, so each loop pauses long
        # enough to read the lead item before scrolling resumes.
        self._detail_dwell: Dict[str, int] = {}
        # Snapshot of the previous ticker text per project — used by
        # `_populate` to detect content changes so a new job lands at
        # offset 0 with a fresh dwell, not whatever offset the old
        # ticker happened to be on.
        self._detail_text_seen: Dict[str, str] = {}
        # Column key for the Detail cell — captured at on_mount.
        self._detail_col_key = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ServerStatusBar(id="server-status")
        with Horizontal(id="list-body"):
            yield DataTable(id="projects-table", cursor_type="row")
            with Vertical(id="project-summary"):
                yield Static("[dim]Highlight a project to see its summary.[/dim]",
                             id="summary-meta")
                yield Static("", id="summary-ribbon")
                yield Static("", id="summary-stats")
        yield Footer()

    # Marquee cadence + window. ~7 cells/sec reads as moving without
    # overwhelming — slow enough to track a glyph, fast enough not to
    # feel stalled. _DETAIL_WIDTH is the visible window; the underlying
    # ticker can be much longer and scrolls into view a column at a
    # time. _DETAIL_GAP is whitespace inserted between the end of the
    # ticker and the start so the loop "feels" continuous rather than
    # snapping.
    _DETAIL_TICK_SECONDS = 0.15
    _DETAIL_WIDTH = 36
    _DETAIL_GAP = 6
    # Dwell ticks the marquee holds at offset=0 before scrolling begins,
    # so the start of the ticker (most important info: what's running)
    # gets ~2.5s of stable reading time. Re-applies on every wrap.
    _DETAIL_DWELL_TICKS = 17

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # Sorted by recency by default; the `Updated` column makes the
        # ordering legible (relative timestamps — "2h", "3d", "5w").
        # `Detail` is a per-row carousel — content cycles every
        # `_DETAIL_TICK_SECONDS` so multi-job projects can show every
        # active piece without a wide column.
        keys = table.add_columns(
            "Name", "Target", "Status", "Updated", "Detail"
        )
        self._detail_col_key = keys[-1] if keys else None
        self._populate()
        self.set_interval(self._DETAIL_TICK_SECONDS, self._tick_details)

    @staticmethod
    def _project_mtime(proj: BeakProject) -> float:
        """Manifest mtime — proxy for "most recent activity".

        Every beak code path that records progress (job submit / pull
        completion / reset / rename) goes through `BeakProject.write`
        or `mutate()`, both of which advance the manifest TOML's mtime.
        Sorting by this descending puts whichever project the user
        last touched (or the system last completed work for) at the
        top of the list.
        """
        try:
            return proj.manifest_path.stat().st_mtime
        except OSError:
            return 0.0

    def _populate(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        projects = BeakProject.list_projects()
        # Sort: most recent activity first. Stable on ties.
        projects.sort(key=self._project_mtime, reverse=True)
        self._projects = projects
        if not self._projects:
            self._update_summary(None)
            self.notify(
                "No projects yet. Create one with: beak project init <name> --uniprot <id>",
                timeout=8,
            )
            return

        live_names = set()
        for proj in self._projects:
            row = self._row_for(proj)
            updated = _relative_age(self._project_mtime(proj))
            text, color = self._compute_detail_ticker(proj)
            self._detail_text[proj.name] = text
            self._detail_color[proj.name] = color
            # Reset offset + dwell when the ticker text changes so a
            # new in-flight job lands at the start of the visible
            # window with a full reading-pause before scrolling.
            prev = getattr(self, "_detail_text_seen", {}).get(proj.name)
            if prev != text:
                self._detail_offset[proj.name] = 0
                self._detail_dwell[proj.name] = self._DETAIL_DWELL_TICKS
            else:
                self._detail_offset.setdefault(proj.name, 0)
                self._detail_dwell.setdefault(proj.name, self._DETAIL_DWELL_TICKS)
            detail = self._render_marquee(proj.name)
            table.add_row(
                row["name"], row["target"], row["status"], updated, detail,
                key=row["name"],
            )
            live_names.add(proj.name)
        # Stash the current ticker strings so the next _populate can
        # detect content changes (new job, completed job, etc.) and
        # reset the offset to 0 with a fresh dwell.
        self._detail_text_seen = dict(self._detail_text)
        # Drop cache entries for projects that no longer exist.
        for stale in list(self._row_cache.keys() - live_names):
            self._row_cache.pop(stale, None)
            self._detail_text.pop(stale, None)
            self._detail_color.pop(stale, None)
            self._detail_offset.pop(stale, None)
            self._detail_dwell.pop(stale, None)
        # Seed the summary with the first row.
        if self._projects:
            self._update_summary(self._projects[0])

    def _compute_detail_ticker(self, proj: BeakProject) -> tuple:
        """Build the per-project ticker string + color.

        Returns `(plain_text, color_tag)`. The text is plain (no inline
        markup) so we can slice it for the marquee window without
        breaking rich's tag pairs. Color is applied to the whole
        visible window — yellow when any job is in flight, dim when
        the project is settled.
        """
        m = proj.manifest()
        sets = (m.get("homologs") or {}).get("sets") or []
        active_set_name = (m.get("homologs") or {}).get("active") or "default"

        chunks: List[str] = []
        any_inflight = False

        for s in sets:
            remote = s.get("remote") or {}
            sname = s.get("name") or "?"
            tag = "" if sname == active_set_name else f" {sname}"
            if remote.get("search_job_id") and not s.get("n_homologs"):
                db = remote.get("search_database") or "?"
                chunks.append(f"⊳ search {db}{tag}")
                any_inflight = True
            if (remote.get("align_job_id")
                    and s.get("n_homologs")
                    and not s.get("n_aligned")):
                chunks.append(f"⊳ aligning{tag}")
                any_inflight = True

        tax = m.get("taxonomy") or {}
        if (tax.get("remote") or {}).get("job_id") and not tax.get("n_assigned"):
            chunks.append("⊳ taxonomy")
            any_inflight = True

        for e in (m.get("embeddings") or {}).get("sets") or []:
            remote = e.get("remote") or {}
            if remote.get("job_id") and not e.get("n_embeddings"):
                model = (e.get("model") or "?").split("/")[-1]
                src = e.get("source_homologs_set") or "?"
                tag = "" if src == active_set_name else f" / {src}"
                chunks.append(f"⊳ embed {model}{tag}")
                any_inflight = True

        # Settled summary always tails the in-flight items so a project
        # with no jobs running shows a stationary "451 hits · 2 models"
        # in the column rather than nothing.
        active_set = next(
            (s for s in sets if s.get("name") == active_set_name), None,
        )
        settled: List[str] = []
        if active_set and active_set.get("n_homologs"):
            n = active_set["n_homologs"]
            n_aligned = active_set.get("n_aligned") or 0
            settled.append(f"{n_aligned:,}/{n:,} aln" if n_aligned else f"{n:,} hits")
        n_models = sum(
            1 for e in (m.get("embeddings") or {}).get("sets") or []
            if e.get("n_embeddings")
        )
        if n_models:
            settled.append(f"{n_models} model{'s' if n_models > 1 else ''}")
        n_exp = len(m.get("experiments") or [])
        if n_exp:
            settled.append(f"{n_exp} exp")
        if settled:
            chunks.append(" · ".join(settled))

        if not chunks:
            return ("—", "dim")

        text = "   ·   ".join(chunks)
        # Palette parity with the status column: amber when work is in
        # flight (matches the in-flight pill / "embed"/"running"
        # statuses), dim when the project is settled.
        color = _BEAK_AMBER if any_inflight else "dim"
        return (text, color)

    def _render_marquee(self, name: str) -> str:
        """Return the colored window slice for `name`'s ticker."""
        text = self._detail_text.get(name) or ""
        color = self._detail_color.get(name) or "dim"
        if not text:
            return ""
        # If the text fits without wrapping, render it static — no
        # point scrolling something that already fits.
        if len(text) <= self._DETAIL_WIDTH:
            return f"[{color}]{text}[/{color}]"
        offset = self._detail_offset.get(name, 0)
        loop = text + " " * self._DETAIL_GAP + text
        slice_ = loop[offset:offset + self._DETAIL_WIDTH]
        return f"[{color}]{slice_}[/{color}]"

    def _tick_details(self) -> None:
        """Advance each long-text project's offset by one cell.

        The marquee dwells at offset=0 for `_DETAIL_DWELL_TICKS` ticks
        before the slide starts — so the lead item (whatever's in
        flight) gets a couple seconds of stable reading time. After
        the offset wraps back to 0, the dwell counter resets and the
        pause repeats. No animation work happens for tickers whose
        text already fits the column width.
        """
        if not self._detail_col_key or not self._projects:
            return
        try:
            table = self.query_one(DataTable)
        except Exception:
            return
        for proj in self._projects:
            text = self._detail_text.get(proj.name) or ""
            if len(text) <= self._DETAIL_WIDTH:
                continue

            # Burn dwell ticks at offset 0 — no cell update needed
            # since the rendered slice doesn't change.
            if self._detail_dwell.get(proj.name, 0) > 0:
                self._detail_dwell[proj.name] -= 1
                continue

            loop_len = len(text) + self._DETAIL_GAP
            offset = (self._detail_offset.get(proj.name, 0) + 1) % loop_len
            self._detail_offset[proj.name] = offset
            # On wrap, re-arm the dwell so the next loop pauses too.
            if offset == 0:
                self._detail_dwell[proj.name] = self._DETAIL_DWELL_TICKS

            try:
                table.update_cell(
                    proj.name,
                    self._detail_col_key,
                    self._render_marquee(proj.name),
                )
            except Exception:
                pass

    def _row_for(self, proj: BeakProject) -> dict:
        """Mtime-cached row data with always-fresh status.

        Manifest-derived fields (name, target_id) are cached by the
        manifest TOML's mtime so 50-project refreshes are cheap. The
        `status` column ignores the cache: a job's lifecycle updates
        `~/.beak/jobs.json` without touching the project manifest, so
        a cached status would go stale. `status_summary` is also
        manifest-cheap (one TOML read + one JSON read), making the
        per-row cost acceptable on every refresh.
        """
        try:
            mtime = proj.manifest_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        cached = self._row_cache.get(proj.name)
        if cached is not None and cached[0] == mtime:
            row = dict(cached[1])
            row["status"] = _format_status(proj.status_summary())
            return row
        m = proj.manifest()
        target = m.get("target", {}) or {}
        target_id = target.get("uniprot_id") or target.get("uniprot_name") or "-"
        status_str = _format_status(proj.status_summary())
        row = {"name": proj.name, "target": target_id, "status": status_str}
        self._row_cache[proj.name] = (mtime, dict(row))
        return row

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "projects-table":
            return
        key = event.row_key.value if event.row_key else None
        if not key:
            return
        proj = next((p for p in self._projects if p.name == str(key)), None)
        self._update_summary(proj)

    def _update_summary(self, proj: Optional[BeakProject]) -> None:
        meta = self.query_one("#summary-meta", Static)
        ribbon = self.query_one("#summary-ribbon", Static)
        stats = self.query_one("#summary-stats", Static)
        if proj is None:
            meta.update("[dim]No project selected.[/dim]")
            ribbon.update("")
            stats.update("")
            return

        m = proj.manifest()
        target = m.get("target", {}) or {}
        proj_meta = m.get("project", {}) or {}

        meta_lines = [f"[bold]{proj.name}[/bold]"]
        if proj_meta.get("description"):
            meta_lines.append(f"[dim]{proj_meta['description']}[/dim]")
        meta_lines.append("")
        for label, key in (
            ("UniProt", "uniprot_id"),
            ("Gene", "gene_name"),
            ("Organism", "organism"),
            ("Length", "length"),
        ):
            val = target.get(key)
            if val is not None:
                if key == "length":
                    val = f"{val} aa"
                meta_lines.append(f"[bold]{label}:[/bold] {val}")
        meta.update("\n".join(meta_lines))

        # Cached AlphaFold preview — never fetches from this list, just
        # reads the on-disk CIF if it's already there.
        ribbon.update(self._render_ribbon_preview(proj, ribbon.size.width, ribbon.size.height))
        stats.update(self._render_layer_summary(proj, m))

    def _render_layer_summary(self, proj: BeakProject, m: dict) -> str:
        rows = []
        active = proj.active_set() or {}
        n_homologs = active.get("n_homologs", 0)
        n_aligned = active.get("n_aligned", 0)
        if n_homologs:
            seg = f"{n_homologs:,} homologs"
            if n_aligned:
                seg += f" / {n_aligned:,} aligned"
            rows.append(f"[green]✓[/green] {seg}")
        else:
            rows.append("[dim]○ no homologs[/dim]")

        # Sum embeddings across every homolog set so the project list
        # reflects total work, not just the active set's slice.
        n_emb = sum(
            (s.get("n_embeddings") or 0) for s in proj.embeddings_sets()
        )
        n_emb_sets = sum(
            1 for s in proj.embeddings_sets() if s.get("n_embeddings")
        )
        if n_emb:
            suffix = f" · {n_emb_sets} sets" if n_emb_sets > 1 else ""
            rows.append(f"[green]✓[/green] {n_emb:,} embeddings{suffix}")
        else:
            rows.append("[dim]○ no embeddings[/dim]")

        struct_dir = proj.path / "structures"
        n_struct = (
            len(list(struct_dir.glob("*.cif"))) if struct_dir.exists() else 0
        )
        if n_struct:
            rows.append(f"[green]✓[/green] {n_struct} structure file(s)")
        else:
            rows.append("[dim]○ no structures[/dim]")

        exps = m.get("experiments") or []
        if exps:
            names = ", ".join(
                e.get("name") if isinstance(e, dict) else str(e) for e in exps[:3]
            )
            tail = "" if len(exps) <= 3 else f" (+{len(exps) - 3})"
            rows.append(f"[green]✓[/green] {len(exps)} experiment(s): {names}{tail}")
        else:
            rows.append("[dim]○ no experiments[/dim]")

        rows.append("")
        rows.append(f"[dim]On disk: {_human_size(proj.cached_size())}[/dim]")
        rows.append(f"[dim]Created: {_format_dt(proj_meta_get(m, 'created_at'))}[/dim]")
        return "\n".join(rows)

    def _render_ribbon_preview(self, proj: BeakProject, width: int, height: int) -> str:
        target = (proj.manifest().get("target") or {})
        uniprot = target.get("uniprot_id")
        if not uniprot:
            return "[dim]No structure preview (project has no UniProt ID).[/dim]"
        cif_path = proj.path / "structures" / f"{uniprot}_AF.cif"
        if not cif_path.exists():
            return (
                "[dim]No cached structure. Open the project to fetch the "
                "AlphaFold model.[/dim]"
            )
        try:
            from ..structure import load_ca_coords, render_structure
            coords, plddt = load_ca_coords(cif_path)
            # PCA-align so the long axis is roughly horizontal — same trick
            # the StructureView uses on first load.
            centered = coords - coords.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            if np.linalg.det(vt) < 0:
                vt[2] *= -1
            coords = centered @ vt.T
        except Exception as e:  # noqa: BLE001
            return f"[dim red]Structure preview failed: {e}[/dim red]"
        # Subtract a small margin to match the panel border drawing.
        w = max(8, width - 2)
        h = max(6, height - 2)
        return render_structure(
            coords, plddt, w, h,
            angle_y=0.0, angle_x=0.0,
            color_mode="plddt", midpoint=50.0,
            view_mode="trace",
        )

    def action_refresh(self) -> None:
        self._populate()
        self.notify("Refreshed")

    def action_settings(self) -> None:
        from .remote_setup import RemoteSetupModal
        self.app.push_screen(RemoteSetupModal())

    def action_new_project(self) -> None:
        from .new_project import NewProjectModal
        self.app.push_screen(NewProjectModal(), self._on_project_created)

    def _on_project_created(self, name) -> None:
        if not name:
            return
        self.notify(f"Created project '{name}'", timeout=4)
        self._populate()

    def action_rename_project(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is None or not self._projects:
            return
        try:
            row_idx = int(table.cursor_row)
        except (TypeError, ValueError):
            return
        if not (0 <= row_idx < len(self._projects)):
            return
        proj = self._projects[row_idx]
        from .rename_project import RenameProjectModal
        self.app.push_screen(RenameProjectModal(proj), self._on_project_renamed)

    def _on_project_renamed(self, new_name) -> None:
        if not new_name:
            return
        self.notify(f"Renamed to '{new_name}'", timeout=4)
        self._populate()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from .detail import ProjectDetailScreen

        if event.row_key is None or event.row_key.value is None:
            return
        name = str(event.row_key.value)
        try:
            proj = BeakProject.load(name)
        except Exception as e:  # noqa: BLE001 — surface any load failure to user
            self.notify(f"Failed to open '{name}': {e}", severity="error")
            return
        self.app.push_screen(ProjectDetailScreen(proj))


def proj_meta_get(m: dict, key: str):
    return (m.get("project") or {}).get(key)
