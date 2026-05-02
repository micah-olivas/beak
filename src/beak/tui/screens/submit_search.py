"""Modal screen for submitting an MMseqs2 search for a project.

The "Set name" field decides which homolog set the resulting hits land
in. Picking an existing set name re-runs into that set (overwriting on
completion); a new name creates a new set and makes it active.
"""

import re
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ...project import BeakProject


_SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


class SubmitSearchModal(ModalScreen[Optional[str]]):
    """Submit an MMseqs2 search for the project's target sequence.

    Dismisses with the job_id on success, or None if cancelled / failed.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._submitting = False

    def compose(self) -> ComposeResult:
        from ...remote.search import MMseqsSearch

        dbs = list(MMseqsSearch.AVAILABLE_DBS.keys())
        presets = list(MMseqsSearch.PRESETS.keys())
        default_db = "uniref90" if "uniref90" in dbs else dbs[0]

        # Suggest a fresh set name if existing ones are taken; otherwise
        # default to "default" so first-time users don't see jargon.
        existing = {s.get("name") for s in self._project.homologs_sets()}
        default_set = "default" if "default" not in existing else f"set_{len(existing) + 1}"

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Submit MMseqs2 search · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("[dim]Run on the configured remote server[/dim]")
            yield Label("")

            yield Label("Set name (creates a new homolog set when novel)")
            yield Input(value=default_set, id="set-name")
            yield Label("Database")
            yield Select(
                [(db, db) for db in dbs],
                value=default_db,
                id="db-select",
                allow_blank=False,
            )
            yield Label("Preset")
            yield Select(
                [(p, p) for p in presets],
                value="default",
                id="preset-select",
                allow_blank=False,
            )
            yield Label("Job name (optional)")
            yield Input(
                placeholder=f"{self._project.name}_search_<auto>",
                id="job-name",
            )
            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Submit", id="submit-btn", variant="primary")

    def action_cancel(self) -> None:
        if self._submitting:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn" and not self._submitting:
            self._do_submit()

    @work(thread=True, exclusive=True, group="search-submit")
    def _do_submit(self) -> None:
        set_name = self.query_one("#set-name", Input).value.strip() or "default"
        if not _SET_NAME_RE.match(set_name):
            self.app.call_from_thread(
                self._set_status,
                "[red]Set name: letters/digits/'_'/'-', start with letter or digit.[/red]",
            )
            return

        db = self.query_one("#db-select", Select).value
        preset = self.query_one("#preset-select", Select).value
        job_name = self.query_one("#job-name", Input).value.strip()
        if not job_name:
            job_name = f"{self._project.name}_{set_name}_search"

        self._submitting = True
        self.app.call_from_thread(
            self._set_status, "[dim]Submitting (this can take a few seconds)…[/dim]"
        )

        try:
            from ...remote.search import MMseqsSearch
            mgr = MMseqsSearch()
            kwargs = {"database": db, "job_name": job_name}
            if preset and preset != "default":
                kwargs["preset"] = preset
            job_id = mgr.submit(str(self._project.target_sequence_path), **kwargs)

            # Register the set (or update an existing one with the same
            # name) and make it active so all downstream pulls land here.
            self._project.add_homolog_set(set_name, remote={
                "search_job_id": job_id,
                "search_database": db,
            })
            self._project.set_active_set(set_name)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_status, f"[red]{type(e).__name__}: {e}[/red]"
            )
            self._submitting = False
            return

        self.app.call_from_thread(self.dismiss, job_id)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
