"""Modal for configuring + submitting an ESM embedding job."""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ...project import BeakProject


class SubmitEmbedModal(ModalScreen[Optional[str]]):
    """Pick model + output options for an ESM embedding job."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._submitting = False

    def compose(self) -> ComposeResult:
        from ...remote.embeddings import ESMEmbeddings

        models = list(ESMEmbeddings.AVAILABLE_MODELS.keys())
        default_model = "esm2_t33_650M_UR50D" if "esm2_t33_650M_UR50D" in models else models[0]

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Submit ESM embeddings · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("[dim]Runs on the configured remote GPU[/dim]")
            yield Label("")

            yield Label("Model")
            yield Select(
                [(m, m) for m in models],
                value=default_model,
                id="model-select",
                allow_blank=False,
            )
            yield Label("Layer (e.g. -1 for last; comma-sep for multiple)")
            yield Input(value="-1", id="layer-input")
            yield Label("Job name (optional)")
            yield Input(
                placeholder=f"{self._project.name}_embed",
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

    @work(thread=True, exclusive=True, group="embed-submit")
    def _do_submit(self) -> None:
        model = str(self.query_one("#model-select", Select).value)
        layers_str = self.query_one("#layer-input", Input).value.strip() or "-1"
        try:
            layers = [int(x) for x in layers_str.split(",") if x.strip()]
        except ValueError:
            self.app.call_from_thread(
                self._set_status, "[red]Layer must be int(s) — e.g. -1 or 30,33[/red]"
            )
            return
        job_name = self.query_one("#job-name", Input).value.strip()
        if not job_name:
            job_name = f"{self._project.name}_embed"

        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self.app.call_from_thread(
                self._set_status,
                "[red]No hits.fasta — run a search first.[/red]",
            )
            return

        self._submitting = True
        self.app.call_from_thread(
            self._set_status, "[dim]Submitting (a few seconds)…[/dim]"
        )

        mgr = None
        try:
            from ...remote.embeddings import ESMEmbeddings
            mgr = ESMEmbeddings()
            job_id = mgr.submit(
                str(hits_fasta),
                model=model,
                job_name=job_name,
                repr_layers=layers,
            )
            with self._project.mutate() as manifest:
                emb = manifest.setdefault("embeddings", {})
                remote = emb.setdefault("remote", {})
                remote["job_id"] = job_id
                emb["model"] = model
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
