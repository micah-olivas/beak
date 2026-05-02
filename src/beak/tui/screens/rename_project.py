"""Modal for renaming a beak project — moves the directory + manifest."""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ...project import BeakProject, BeakProjectError


class RenameProjectModal(ModalScreen[Optional[str]]):
    """Edit a project's name. Dismisses with the new name on success."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label("[bold]Rename project[/bold]", id="modal-title")
            yield Label(
                f"[dim]Current name: {self._project.name}[/dim]"
            )
            yield Label("")
            yield Label("New name")
            yield Input(value=self._project.name, id="name-input")
            yield Label("", id="status-line")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Rename", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name-input":
            self._submit()

    def _submit(self) -> None:
        new_name = self.query_one("#name-input", Input).value.strip()
        if not new_name:
            self._status("[red]Name is required.[/red]")
            return
        try:
            self._project.rename(new_name)
        except BeakProjectError as e:
            self._status(f"[red]{e}[/red]")
            return
        self.dismiss(new_name)

    def _status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
