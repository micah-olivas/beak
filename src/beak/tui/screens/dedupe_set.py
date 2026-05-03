"""Confirmation modal for deduplicating a homolog set's sequences.

Apply triggers `BeakProject.dedupe_homolog_set`, which streams the FASTA
once, drops exact-sequence duplicates, and clears the existing
alignment for this set (since the old alignment was built over the
redundant FASTA).
"""

from typing import Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from ...project import BeakProject, BeakProjectError


class DedupeSetModal(ModalScreen[Optional[Tuple[int, int]]]):
    """Confirm dedup of a set; dismisses with (n_kept, n_dropped) on apply."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject, set_name: str) -> None:
        super().__init__()
        self._project = project
        self._set_name = set_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Deduplicate '{self._set_name}'[/bold]",
                id="modal-title",
            )
            yield Label(
                "[dim]Streams sequences.fasta and keeps the first occurrence "
                "of each unique sequence (case-insensitive, exact match). "
                "If anything is dropped, the existing alignment for this "
                "set is cleared so you can re-align the smaller FASTA.[/dim]"
            )
            yield Label("")
            yield Label("", id="status-line")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Apply", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self._submit()

    def _submit(self) -> None:
        try:
            n_kept, n_dropped = self._project.dedupe_homolog_set(self._set_name)
        except BeakProjectError as e:
            self._status(f"[red]{e}[/red]")
            return
        self.dismiss((n_kept, n_dropped))

    def _status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
