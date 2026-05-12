"""Modal for editing a beak project's name and description."""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ...project import BeakProject, BeakProjectError


class RenameProjectModal(ModalScreen[Optional[str]]):
    """Edit a project's name and description.

    Dismisses with the (possibly unchanged) project name on success, or
    None on cancel. A non-None result means *something* was saved, so
    callers should refresh whatever displays project metadata.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project

    def compose(self) -> ComposeResult:
        proj_meta = self._project.manifest().get("project", {}) or {}
        desc = proj_meta.get("description", "") or ""
        with Vertical(id="modal-body"):
            yield Label("[bold]Edit project[/bold]", id="modal-title")
            yield Label("")
            yield Label("Name")
            yield Input(value=self._project.name, id="name-input")
            yield Label("Description")
            yield Input(
                value=desc,
                placeholder="optional — one-line summary",
                id="desc-input",
            )
            yield Label("", id="status-line")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Save", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name-input":
            self.query_one("#desc-input", Input).focus()
        elif event.input.id == "desc-input":
            self._submit()

    def _submit(self) -> None:
        new_name = self.query_one("#name-input", Input).value.strip()
        new_desc = self.query_one("#desc-input", Input).value.strip()
        if not new_name:
            self._status("[red]Name is required.[/red]")
            return
        if new_name != self._project.name:
            try:
                self._project.rename(new_name)
            except BeakProjectError as e:
                self._status(f"[red]{e}[/red]")
                return
        try:
            with self._project.mutate() as m:
                m.setdefault("project", {})["description"] = new_desc
        except Exception as e:  # noqa: BLE001
            self._status(f"[red]Could not save description: {e}[/red]")
            return
        self.dismiss(self._project.name)

    def _status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
