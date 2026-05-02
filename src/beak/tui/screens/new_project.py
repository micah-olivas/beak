"""Modal for creating a new beak project from the project list screen.

Matches the surface of `beak project init` — name, description, and
exactly one of `--uniprot` / `--sequence`. UniProt mode fetches the
sequence + metadata from the REST API; sequence-file mode reads a
local FASTA. Either way we end up with a populated `target/` directory
and a fresh manifest.
"""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class NewProjectModal(ModalScreen[Optional[str]]):
    """Capture name + target source, dismiss with the new project name."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label("[bold]New project[/bold]", id="modal-title")
            yield Label(
                "[dim]Create a target-centric project. "
                "Pick one of UniProt accession OR a local FASTA.[/dim]"
            )
            yield Label("")

            yield Label("Project name")
            yield Input(placeholder="e.g. egfr", id="name-input")
            yield Label("Description (optional)")
            yield Input(placeholder="One-line description", id="desc-input")
            yield Label("")
            yield Label("UniProt accession")
            yield Input(placeholder="e.g. P00533", id="uniprot-input")
            yield Label("[dim]— or —[/dim]")
            yield Label("Local FASTA path")
            yield Input(placeholder="/path/to/target.fasta", id="fasta-input")

            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Create", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self._create()

    def _create(self) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        desc = self.query_one("#desc-input", Input).value.strip()
        uniprot = self.query_one("#uniprot-input", Input).value.strip() or None
        fasta = self.query_one("#fasta-input", Input).value.strip() or None

        if not name:
            self._set_status("[red]Project name is required.[/red]")
            return
        if bool(uniprot) == bool(fasta):
            self._set_status(
                "[red]Provide exactly one of UniProt accession or FASTA path.[/red]"
            )
            return
        if fasta and not Path(fasta).expanduser().exists():
            self._set_status(f"[red]FASTA not found: {fasta}[/red]")
            return

        try:
            from ...project import BeakProject, BeakProjectError
            BeakProject.init(
                name=name,
                uniprot_id=uniprot,
                sequence_file=str(Path(fasta).expanduser()) if fasta else None,
                description=desc,
            )
        except BeakProjectError as e:
            self._set_status(f"[red]{e}[/red]")
            return
        except Exception as e:  # noqa: BLE001
            self._set_status(f"[red]{type(e).__name__}: {e}[/red]")
            return

        self.dismiss(name)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
