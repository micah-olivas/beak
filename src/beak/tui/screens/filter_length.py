"""Modal for trimming a homolog set by sequence length range.

Applying overwrites `sequences.fasta` in place and clears the existing
alignment + conservation cache for that set, since they no longer
correspond to the FASTA on disk.
"""

from typing import Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ...project import BeakProject, BeakProjectError


class FilterLengthModal(ModalScreen[Optional[Tuple[int, int, int]]]):
    """Edit min/max length for a set; dismisses with (min, max, n_kept)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        project: BeakProject,
        set_name: str,
        current_min: int,
        current_max: int,
    ) -> None:
        super().__init__()
        self._project = project
        self._set_name = set_name
        self._cur_min = current_min
        self._cur_max = current_max

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Filter '{self._set_name}' by length[/bold]",
                id="modal-title",
            )
            yield Label(
                f"[dim]Current range: {self._cur_min}–{self._cur_max} aa[/dim]"
            )
            yield Label(
                "[dim]Sequences outside the range will be dropped, "
                "and the existing alignment for this set will be cleared.[/dim]"
            )
            yield Label("")
            yield Label("Min length")
            yield Input(
                value=str(self._cur_min),
                id="min-input",
                type="integer",
            )
            yield Label("Max length")
            yield Input(
                value=str(self._cur_max),
                id="max-input",
                type="integer",
            )
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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("min-input", "max-input"):
            self._submit()

    def _submit(self) -> None:
        try:
            mn = int(self.query_one("#min-input", Input).value or "0")
            mx = int(self.query_one("#max-input", Input).value or "0")
        except ValueError:
            self._status("[red]Min and max must be integers.[/red]")
            return
        if mn < 0 or mx < 0:
            self._status("[red]Lengths must be non-negative.[/red]")
            return
        if mn > mx:
            self._status("[red]Min must be ≤ max.[/red]")
            return
        try:
            n_kept = self._project.filter_homolog_set_by_length(
                self._set_name, mn, mx
            )
        except BeakProjectError as e:
            self._status(f"[red]{e}[/red]")
            return
        self.dismiss((mn, mx, n_kept))

    def _status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
