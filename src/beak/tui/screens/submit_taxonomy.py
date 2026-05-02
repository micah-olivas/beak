"""Modal for submitting an MMseqs2 LCA taxonomy job for the project's hits.

Used when sequences don't have UniProt accessions (BFD, metagenomic) or
when the user wants a fresh remote LCA assignment alongside the auto
UniProt-REST taxonomy.
"""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ...project import BeakProject


class SubmitTaxonomyModal(ModalScreen[Optional[str]]):
    """Pick database, submit MMseqs2 taxonomy on the project's hits.fasta."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._submitting = False

    def compose(self) -> ComposeResult:
        from ...remote.taxonomy import MMseqsTaxonomy
        dbs = list(MMseqsTaxonomy.AVAILABLE_DBS.keys())
        default_db = "uniprotkb" if "uniprotkb" in dbs else dbs[0]

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Submit taxonomy · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("[dim]MMseqs2 LCA against a remote taxonomic DB[/dim]")
            yield Label("")
            yield Label("Database")
            yield Select(
                [(db, db) for db in dbs],
                value=default_db,
                id="tax-db-select",
                allow_blank=False,
            )
            yield Label("Job name (optional)")
            yield Input(
                placeholder=f"{self._project.name}_tax",
                id="tax-job-name",
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

    @work(thread=True, exclusive=True, group="tax-submit")
    def _do_submit(self) -> None:
        db = str(self.query_one("#tax-db-select", Select).value)
        job_name = self.query_one("#tax-job-name", Input).value.strip()
        if not job_name:
            job_name = f"{self._project.name}_tax"

        hits = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits.exists():
            self.app.call_from_thread(
                self._set_status, "[red]No hits.fasta — run a search first.[/red]"
            )
            return

        self._submitting = True
        self.app.call_from_thread(
            self._set_status, "[dim]Submitting (a few seconds)…[/dim]"
        )
        mgr = None
        try:
            from ...remote.taxonomy import MMseqsTaxonomy
            mgr = MMseqsTaxonomy()
            job_id = mgr.submit(str(hits), database=db, job_name=job_name)
            with self._project.mutate() as manifest:
                tax = manifest.setdefault("taxonomy", {})
                remote = tax.setdefault("remote", {})
                remote["job_id"] = job_id
                remote["database"] = db
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_status, f"[red]{type(e).__name__}: {e}[/red]"
            )
            self._submitting = False
            return
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass
        self.app.call_from_thread(self.dismiss, job_id)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
