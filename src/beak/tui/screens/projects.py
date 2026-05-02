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


_STATUS_COLORS = {
    "ready":     "green",
    "running":   "yellow",
    "submitted": "cyan",
    "queued":    "cyan",
    "completed": "green",
    "failed":    "red",
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

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # Trimmed columns — size/length/created move to the right-pane summary.
        table.add_columns("Name", "Target", "Status")
        self._populate()

    def _populate(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self._projects = BeakProject.list_projects()
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
            table.add_row(
                row["name"], row["target"], row["status"], key=row["name"]
            )
            live_names.add(proj.name)
        # Drop cache entries for projects that no longer exist.
        for stale in list(self._row_cache.keys() - live_names):
            self._row_cache.pop(stale, None)
        # Seed the summary with the first row.
        if self._projects:
            self._update_summary(self._projects[0])

    def _row_for(self, proj: BeakProject) -> dict:
        """Mtime-cached row data. Re-reads the manifest only when the
        TOML file's mtime has advanced since the last cached read."""
        try:
            mtime = proj.manifest_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        cached = self._row_cache.get(proj.name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        m = proj.manifest()
        target = m.get("target", {}) or {}
        target_id = target.get("uniprot_id") or target.get("uniprot_name") or "-"
        # `status_summary` reads the manifest itself; we can't easily
        # avoid that second read without refactoring the API, but it's
        # cheap and only fires on the slow path (cache miss).
        status_str = _format_status(proj.status_summary())
        row = {"name": proj.name, "target": target_id, "status": status_str}
        self._row_cache[proj.name] = (mtime, row)
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

        emb = m.get("embeddings") or {}
        n_emb = emb.get("n_embeddings", 0)
        if n_emb:
            rows.append(f"[green]✓[/green] {n_emb:,} embeddings")
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
