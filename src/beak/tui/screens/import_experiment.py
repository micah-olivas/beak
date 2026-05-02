"""Modal to import a CSV of experimental measurements into a project.

Two-stage flow:
    1. Enter the CSV path and click `Preview`.
    2. The first 5 rows render in a DataTable. The pos/pert column
       dropdowns are populated from the actual header. Numeric columns
       get auto-suggested as property columns. Hit `Import` to copy.

The CSV is copied to `experiments/<name>/raw.csv` and the manifest
gets a new `[[experiments]]` entry. Re-import with the same name
overwrites idempotently.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Select

from ...project import BeakProject


class ImportExperimentModal(ModalScreen[Optional[str]]):
    """Pick CSV → preview → choose columns → import."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    ImportExperimentModal #csv-preview {
        height: 8;
        margin: 1 0;
    }
    ImportExperimentModal #csv-preview.-empty { display: none; }
    ImportExperimentModal Input { margin-bottom: 1; }
    ImportExperimentModal Select { margin-bottom: 1; }
    ImportExperimentModal #path-row Input { width: 1fr; }
    ImportExperimentModal #path-row Button { width: auto; margin-left: 1; }
    """

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._columns: List[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Import experiment · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("[dim]CSV with one row per (position, perturbation)[/dim]")
            yield Label("")

            yield Label("CSV file path")
            with Horizontal(id="path-row"):
                yield Input(placeholder="/path/to/measurements.csv", id="csv-path")
                yield Button("Preview", id="preview-btn")

            preview = DataTable(id="csv-preview", classes="-empty")
            preview.cursor_type = "none"
            yield preview

            yield Label("Experiment name")
            yield Input(placeholder="e.g. dms_round1", id="exp-name")

            yield Label("Position column")
            yield Select([("(preview first)", "")], value="", id="pos-select",
                         allow_blank=True)
            yield Label("Perturbation column")
            yield Select([("(preview first)", "")], value="", id="pert-select",
                         allow_blank=True)
            yield Label("Property columns (comma-separated)")
            yield Input(placeholder="auto-filled from numeric columns",
                        id="prop-cols")

            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Import", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "preview-btn":
            self._do_preview()
        elif event.button.id == "submit-btn":
            self._do_import()

    # ---- Preview ----

    def _do_preview(self) -> None:
        path = Path(self.query_one("#csv-path", Input).value.strip()).expanduser()
        if not path.exists():
            self._set_status(f"[red]File not found: {path}[/red]")
            return

        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"[red]Could not read CSV: {e}[/red]")
            return

        columns = list(df.columns)
        self._columns = columns

        # Populate the head preview table.
        table = self.query_one("#csv-preview", DataTable)
        table.clear(columns=True)
        for col in columns:
            table.add_column(str(col))
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row.tolist()])
        table.remove_class("-empty")

        # Populate column dropdowns.
        opts = [(c, c) for c in columns]
        pos_sel = self.query_one("#pos-select", Select)
        pert_sel = self.query_one("#pert-select", Select)
        pos_sel.set_options(opts)
        pert_sel.set_options(opts)
        # Auto-detect common column names.
        for cand in ("pos", "position", "Position", "site"):
            if cand in columns:
                pos_sel.value = cand
                break
        for cand in ("sub", "perturbation", "mut", "mutation", "AA"):
            if cand in columns:
                pert_sel.value = cand
                break

        # Auto-fill property columns from numeric dtypes.
        try:
            numeric_cols = list(df.select_dtypes(include="number").columns)
            self.query_one("#prop-cols", Input).value = ",".join(numeric_cols)
        except Exception:
            pass

        self._set_status(f"[green]Loaded {len(columns)} columns from {path.name}[/green]")

    # ---- Import ----

    def _do_import(self) -> None:
        path = Path(self.query_one("#csv-path", Input).value.strip()).expanduser()
        name = self.query_one("#exp-name", Input).value.strip()
        pos_col = str(self.query_one("#pos-select", Select).value or "")
        pert_col = str(self.query_one("#pert-select", Select).value or "")
        prop_cols = [
            c.strip() for c in self.query_one("#prop-cols", Input).value.split(",")
            if c.strip()
        ]

        if not path.exists():
            self._set_status(f"[red]File not found: {path}[/red]")
            return
        if not name:
            self._set_status("[red]Experiment name is required.[/red]")
            return
        if not pos_col or not pert_col:
            self._set_status("[red]Pick position + perturbation columns first (Preview).[/red]")
            return

        # Validate that the selected columns exist in the actual file.
        if self._columns:
            missing = [c for c in (pos_col, pert_col, *prop_cols) if c not in self._columns]
            if missing:
                self._set_status(
                    f"[red]Columns not in CSV: {', '.join(missing)}[/red]"
                )
                return

        try:
            dest_dir = self._project.path / "experiments" / name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "raw.csv"
            shutil.copy2(path, dest)

            manifest = self._project.manifest()
            exps = manifest.setdefault("experiments", [])
            exps[:] = [
                e for e in exps
                if not (isinstance(e, dict) and e.get("name") == name)
            ]
            exps.append({
                "name": name,
                "file": f"experiments/{name}/raw.csv",
                "position_col": pos_col,
                "perturbation_col": pert_col,
                "property_cols": prop_cols,
                "imported_at": datetime.now(),
            })
            self._project.write(manifest)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"[red]Import failed: {e}[/red]")
            return

        self.dismiss(name)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
