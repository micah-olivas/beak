"""Modal for first-time / on-demand remote setup.

Captures the four fields `beak config init` asks for at the CLI:
host, user, SSH key path, remote job directory. Writes them to
`~/.beak/config.toml` under `[connection]` so subsequent SSH-based
features (search, align, embed) Just Work.
"""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class RemoteSetupModal(ModalScreen[Optional[bool]]):
    """Edit connection settings and save to ~/.beak/config.toml."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, first_run: bool = False) -> None:
        super().__init__()
        self._first_run = first_run

    def compose(self) -> ComposeResult:
        from ...config import load_config

        cfg = load_config()
        conn = cfg.get("connection") or {}

        with Vertical(id="modal-body"):
            yield Label("[bold]Remote setup[/bold]", id="modal-title")
            sub = (
                "[dim]Welcome to beak. Tell me where your compute server is.[/dim]"
                if self._first_run
                else "[dim]Edit SSH connection details for the remote server.[/dim]"
            )
            yield Label(sub)
            yield Label("")

            yield Label("Hostname")
            yield Input(
                value=conn.get("host", "") or "",
                placeholder="server.example.com",
                id="host-input",
            )
            yield Label("Username")
            yield Input(
                value=conn.get("user", "") or "",
                placeholder="your-user",
                id="user-input",
            )
            yield Label("SSH key path")
            yield Input(
                value=conn.get("key_path", "") or "~/.ssh/id_ed25519",
                id="key-input",
            )
            yield Label("Remote job directory")
            yield Input(
                value=conn.get("remote_job_dir", "") or "~/beak_jobs",
                id="dir-input",
            )

            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                if not self._first_run:
                    yield Button("Cancel", id="cancel-btn")
                yield Button(
                    "Save" if not self._first_run else "Get started",
                    id="submit-btn", variant="primary",
                )

    def action_cancel(self) -> None:
        # In first-run mode, escape still saves nothing — the user can
        # configure later from the project list via `s`.
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "submit-btn":
            self._save()

    def _save(self) -> None:
        host = self.query_one("#host-input", Input).value.strip()
        user = self.query_one("#user-input", Input).value.strip()
        key = self.query_one("#key-input", Input).value.strip() or "~/.ssh/id_ed25519"
        job_dir = self.query_one("#dir-input", Input).value.strip() or "~/beak_jobs"

        if not host or not user:
            self._set_status("[red]Host and username are required.[/red]")
            return

        from ...config import load_config, save_config

        cfg = load_config()
        cfg["connection"] = {
            "host": host,
            "user": user,
            "key_path": key,
            "remote_job_dir": job_dir,
        }
        try:
            save_config(cfg)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"[red]Save failed: {e}[/red]")
            return

        self.dismiss(True)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
