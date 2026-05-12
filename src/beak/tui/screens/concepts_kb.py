"""InterPLM concepts knowledge-base management modal.

The InterPLM screen launches this with `c`. It surfaces every (model, layer)
combination, shows what's been computed (n_features, n_proteins, date), and
lets the user run / recompute / delete entries — including a fast smoke-test
sample size so a 40-min full run isn't the only option.

Returns a dict via ``dismiss({...})`` describing what the parent should do:
    {"action": "compute", "model": ..., "layer": ..., "n_proteins": ...}
    {"action": "loaded"}                  — a different (model, layer) was activated
    None                                  — user closed
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, RadioButton, RadioSet, Static

from ... import interplm


_SAMPLE_OPTIONS = [
    ("smoke", 100, "smoke", "~2 min"),
    ("medium", 1000, "medium", "~10 min"),
    ("full", 10000, "full", "~30–40 min"),
]


def _humanize_age(iso_ts: str) -> str:
    """Return '5h ago', '3d ago', etc. given an ISO 8601 timestamp."""
    try:
        ts = datetime.fromisoformat(iso_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        s = delta.total_seconds()
        if s < 3600:
            return f"{int(s // 60)}m ago"
        if s < 86400:
            return f"{int(s // 3600)}h ago"
        if s < 86400 * 30:
            return f"{int(s // 86400)}d ago"
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return "—"


class ConceptsKBModal(ModalScreen):
    """Listing of (model, layer) entries with manage/compute/delete affordances."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    DEFAULT_CSS = """
    ConceptsKBModal {
        align: center middle;
    }
    ConceptsKBModal #panel {
        width: 92;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    ConceptsKBModal #title {
        height: 1;
        content-align: left middle;
        color: $text;
    }
    ConceptsKBModal #blurb {
        height: auto;
        padding: 0 0 1 0;
        color: $text-muted;
    }
    ConceptsKBModal #kb-table {
        height: 14;
        margin-bottom: 1;
    }
    ConceptsKBModal #compute-panel {
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    ConceptsKBModal #compute-title {
        height: 1;
        color: $text;
    }
    ConceptsKBModal RadioSet {
        height: auto;
        padding: 0;
    }
    ConceptsKBModal #button-row {
        height: 3;
        align: right middle;
    }
    ConceptsKBModal Button {
        margin-left: 1;
    }
    ConceptsKBModal #status {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        current_model: str,
        current_layer: int,
        running_job: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self._current = (current_model, current_layer)
        self._running_job: Optional[tuple] = running_job  # (model, layer) or None
        self._selected: Optional[tuple] = None  # (model, layer) currently highlighted
        self._rows: list = []  # (model, layer, has_csv, meta_or_None)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(
                "[bold]InterPLM concept knowledge base[/bold]",
                id="title",
            )
            yield Static(
                "[dim]Asymmetric F1 (precision per residue, recall per "
                "domain) against Swiss-Prot annotations.\n"
                "Highlight a row, then Compute / Recompute / Delete. "
                "Smoke test (100 proteins) takes ~2 min — use it to "
                "validate before the full run.[/dim]",
                id="blurb",
            )
            yield DataTable(id="kb-table", cursor_type="row")
            with Vertical(id="compute-panel"):
                yield Static("[bold]Sample size[/bold]", id="compute-title")
                with RadioSet(id="sample-size"):
                    yield RadioButton(
                        "Smoke test — 100 proteins (~2 min)",
                        id="rb-smoke",
                    )
                    yield RadioButton(
                        "Medium — 1,000 proteins (~10 min)",
                        id="rb-medium",
                    )
                    yield RadioButton(
                        "Full — 10,000 proteins (~30–40 min)",
                        id="rb-full",
                        value=True,
                    )
            yield Static("", id="status")
            with Horizontal(id="button-row"):
                yield Button("Activate", id="activate-btn")
                yield Button("Delete", id="delete-btn", variant="error")
                yield Button(
                    "Compute / Recompute",
                    id="compute-btn",
                    variant="primary",
                )
                yield Button("Close", id="close-btn")

    def on_mount(self) -> None:
        table = self.query_one("#kb-table", DataTable)
        table.add_columns(
            "Model", "Layer", "Status", "Features", "Proteins", "Computed",
        )
        self._rebuild_rows()
        # Pre-select the row that matches what the InterPLM screen had open.
        for i, (m, l, _has, _meta) in enumerate(self._rows):
            if (m, l) == self._current:
                table.move_cursor(row=i)
                self._selected = (m, l)
                break
        self._update_status()
        self._sync_compute_button()
        # Refresh rows on a 2s tick so the user sees live progress in the
        # status column while a remote job is running. The InterPLM screen
        # is occluded by this modal, so this is the user's only progress
        # surface during compute.
        self.set_interval(2.0, self._on_tick)

    def _on_tick(self) -> None:
        self._rebuild_rows()
        self._sync_compute_button()
        self._update_status()

    def _any_job_running(self) -> bool:
        return any(
            rec.get("status") == "running"
            for rec in interplm.get_active_concepts_jobs().values()
        )

    def _sync_compute_button(self) -> None:
        """Disable Compute while any job is running — duplicate submissions
        would race the in-flight remote process and overwrite its CSV.
        """
        try:
            btn = self.query_one("#compute-btn", Button)
        except Exception:
            return
        if self._any_job_running():
            btn.disabled = True
            btn.label = "Job running — wait"
        else:
            btn.disabled = False
            btn.label = "Compute / Recompute"

    def _rebuild_rows(self) -> None:
        table = self.query_one("#kb-table", DataTable)
        table.clear()
        self._rows = []
        # Pull live running state from globals — the `running_job` ctor
        # arg is only authoritative at modal-open time. A job started
        # *while the modal is open* (impossible today, but cheap to be
        # correct about) wouldn't show up otherwise.
        active = interplm.get_active_concepts_jobs()
        running_set = {
            (m, l) for (m, l), rec in active.items()
            if rec.get("status") == "running"
        }
        for model_key, cfg in interplm._MODELS.items():
            for layer in cfg["layers"]:
                has_csv = interplm.has_concepts(layer, model_key)
                meta = interplm.load_concepts_meta(layer, model_key) if has_csv else None
                self._rows.append((model_key, layer, has_csv, meta))
                is_running = (model_key, layer) in running_set
                if is_running:
                    rec = active.get((model_key, layer), {})
                    short_msg = (rec.get("message") or "running")[:24]
                    status = f"[yellow]⟳ {short_msg}[/yellow]"
                    n_feat = "—"
                    n_prot = "—"
                    age = "—"
                elif has_csv and meta:
                    status = "[green]✓[/green] computed"
                    n_feat = f"{meta.get('n_features_annotated', '—'):>6}"
                    n_prot = f"{meta.get('n_proteins_requested', '—'):>6}"
                    age = _humanize_age(meta.get("computed_at", ""))
                elif has_csv:
                    status = "[green]✓[/green] computed (legacy)"
                    n_feat = "?"
                    n_prot = "?"
                    age = "—"
                else:
                    status = "[dim]not computed[/dim]"
                    n_feat = "—"
                    n_prot = "—"
                    age = "—"
                table.add_row(
                    cfg["label"], str(layer), status, n_feat, n_prot, age,
                    key=f"{model_key}|{layer}",
                )

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted,
    ) -> None:
        idx = event.cursor_row
        if idx is None or idx < 0 or idx >= len(self._rows):
            return
        m, l, _has, _meta = self._rows[idx]
        self._selected = (m, l)
        self._update_status()

    def _update_status(self) -> None:
        try:
            status = self.query_one("#status", Static)
        except Exception:
            return
        if not self._selected:
            status.update("")
            return
        m, l = self._selected
        cfg = interplm._MODELS.get(m, {})
        meta = interplm.load_concepts_meta(l, m)
        if meta:
            status.update(
                f"[dim]{cfg.get('label', m)} · layer {l} · "
                f"v{meta.get('methodology_version', '?')} · "
                f"F1≥{meta.get('f1_min', '?')} · "
                f"{meta.get('n_proteins_requested', '?')} proteins[/dim]"
            )
        else:
            status.update(
                f"[dim]{cfg.get('label', m)} · layer {l} · not yet computed[/dim]"
            )

    def _selected_n_proteins(self) -> int:
        rs = self.query_one("#sample-size", RadioSet)
        pressed = rs.pressed_button
        if pressed is None:
            return 10000
        return {"rb-smoke": 100, "rb-medium": 1000, "rb-full": 10000}.get(
            pressed.id or "", 10000,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "close-btn":
            self.action_cancel()
        elif bid == "compute-btn":
            self._compute_or_recompute()
        elif bid == "delete-btn":
            self._delete_selected()
        elif bid == "activate-btn":
            self._activate_selected()

    def _compute_or_recompute(self) -> None:
        if not self._selected:
            self.notify("Highlight a row first.", severity="warning", timeout=4)
            return
        m, l = self._selected
        n = self._selected_n_proteins()
        self.dismiss({
            "action": "compute",
            "model": m,
            "layer": l,
            "n_proteins": n,
        })

    def _delete_selected(self) -> None:
        if not self._selected:
            return
        m, l = self._selected
        if not interplm.has_concepts(l, m):
            self.notify("Nothing to delete for that entry.", timeout=4)
            return
        interplm.delete_concepts(l, m)
        self.notify(f"Deleted concepts: {m} layer {l}", timeout=4)
        self._rebuild_rows()
        self._update_status()

    def _activate_selected(self) -> None:
        """Acknowledge selection — parent will reload the chosen entry."""
        if not self._selected:
            return
        m, l = self._selected
        if not interplm.has_concepts(l, m):
            self.notify(
                "No concepts file for that entry — Compute first.",
                severity="warning", timeout=4,
            )
            return
        self.dismiss({"action": "loaded", "model": m, "layer": l})

    def action_cancel(self) -> None:
        self.dismiss(None)
