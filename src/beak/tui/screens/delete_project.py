"""Modal for deleting a beak project entirely.

Project deletion is unrecoverable, so the modal requires the user to
*type the project name* before the Delete button is enabled — same
pattern as GitHub repo deletion. Cancels via Esc.

Dismisses with True on confirmed deletion, None on cancel. The list
screen pumps the boolean into a worker that handles remote cleanup +
local rmtree; this modal only confirms the user's intent.
"""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ...project import BeakProject


class DeleteProjectModal(ModalScreen[Optional[bool]]):
    """Type-to-confirm modal for project deletion."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        # Snapshot the bits we need to summarize what's about to go
        # away — the user gets one last chance to see it before
        # hitting confirm.
        try:
            self._n_remote_jobs = len(project.remote_job_ids())
        except Exception:  # noqa: BLE001
            self._n_remote_jobs = 0
        try:
            self._size_str = _human_size(project.cached_size())
        except Exception:  # noqa: BLE001
            self._size_str = "?"

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Delete project[/bold] · {self._project.name}",
                id="modal-title",
            )
            yield Label(
                "[red]This is unrecoverable.[/red] The local project "
                "directory will be removed, and any remote scratch "
                "directories for in-flight or completed jobs will be "
                "rm'd from the workstation.",
            )
            yield Label("")
            yield Label(
                f"  [dim]On disk:[/dim]  {self._size_str}",
                id="delete-summary-disk",
            )
            yield Label(
                f"  [dim]Remote scratch dirs:[/dim]  "
                f"{self._n_remote_jobs} job{'s' if self._n_remote_jobs != 1 else ''}",
                id="delete-summary-remote",
            )
            yield Label("")
            yield Label(
                f"Type [bold]{self._project.name}[/bold] to confirm:"
            )
            yield Input(value="", id="confirm-input")
            yield Label("", id="status-line")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                # Starts disabled; the Input.Changed handler enables
                # it once the typed name matches exactly. Warning
                # variant flags the destructive intent visually even
                # while disabled.
                btn = Button(
                    "Delete project", id="submit-btn", variant="warning",
                )
                btn.disabled = True
                yield btn

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self._submit()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "confirm-input":
            return
        match = event.value.strip() == self._project.name
        try:
            self.query_one("#submit-btn", Button).disabled = not match
        except Exception:  # noqa: BLE001
            pass
        if match:
            self._status(
                "[dim]Press Delete to confirm — this can't be undone.[/dim]"
            )
        else:
            self._status("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "confirm-input":
            self._submit()

    def _submit(self) -> None:
        typed = self.query_one("#confirm-input", Input).value.strip()
        if typed != self._project.name:
            self._status(
                "[red]Typed name doesn't match.[/red] Project not deleted."
            )
            return
        self.dismiss(True)

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-line", Label).update(msg)
        except Exception:  # noqa: BLE001
            pass


def _human_size(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"
