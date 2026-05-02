"""Modal showing detailed status of a remote job (parsed log + stages)."""

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


_STATUS_COLOR = {
    "SUBMITTED": "cyan",
    "QUEUED":    "cyan",
    "RUNNING":   "yellow",
    "COMPLETED": "green",
    "FAILED":    "red",
    "CANCELLED": "dim",
    "UNKNOWN":   "dim",
}

_STAGE_ICON = {
    "done":    "[green]✓[/green]",
    "active":  "[yellow]●[/yellow]",
    "pending": "[dim]○[/dim]",
}


class JobStatusModal(ModalScreen):
    """Auto-refreshing detail view for a single remote job."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self._job_id = job_id

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(f"[bold]Job · {self._job_id}[/bold]", id="status-title")
            yield Static("[dim]Loading status…[/dim]", id="status-content")
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close-btn")

    def on_mount(self) -> None:
        self._poll()
        # Refresh while the modal is open — running jobs progress.
        self.set_interval(5.0, self._poll)

    @work(thread=True, exclusive=True, group="job-status-detail")
    def _poll(self) -> None:
        try:
            # `get_manager` looks up `job_type` in ~/.beak/jobs.json and
            # instantiates the right manager class (MMseqsSearch /
            # ClustalAlign / ESMEmbeddings / etc.). Without this, an
            # embedding job's stages are mistakenly rendered against the
            # MMseqs2 LOG_OPERATIONS list ("Counting k-mers", ...).
            from ...cli._common import get_manager
            mgr = get_manager(job_id=self._job_id)
            info = mgr.detailed_status(self._job_id)
        except Exception as e:  # noqa: BLE001
            # `str(e)` for Fabric/Invoke ThreadException is a 50-line dump
            # of every wrapped frame — useless in a modal. Show only the
            # exception class + first line, the rest goes to the log.
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            self.app.call_from_thread(
                self._set_content,
                f"[red]Status check failed:[/red] [dim]{type(e).__name__}[/dim]\n"
                f"[dim]{msg[:200]}[/dim]\n\n"
                f"[dim]The remote job itself may still be running — "
                f"close this and reopen, or check `beak jobs`.[/dim]",
            )
            return
        self.app.call_from_thread(self._render_status, info)

    def _render_status(self, info: dict) -> None:
        status = info.get("status", "UNKNOWN")
        runtime = info.get("runtime", "?")
        stages = info.get("stages") or []
        last_log = info.get("last_log_line") or ""
        progress = info.get("progress") or {}
        color = _STATUS_COLOR.get(status, "dim")

        lines = [
            f"[{color}]●[/{color}] [bold]{status}[/bold]   "
            f"[dim]runtime[/dim] {runtime}",
        ]

        if progress.get("current_operation"):
            lines.append(f"[dim]Stage:[/dim] {progress['current_operation']}")

        if stages:
            lines.append("")
            lines.append("[bold]Stages[/bold]")
            for s in stages:
                icon = _STAGE_ICON.get(s.get("state", ""), "")
                lines.append(f"  {icon} {s.get('label', '')}")

        if last_log:
            lines.append("")
            lines.append("[bold]Latest log[/bold]")
            lines.append(f"[dim]{last_log}[/dim]")

        self._set_content("\n".join(lines))

    def _set_content(self, text: str) -> None:
        self.query_one("#status-content", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
