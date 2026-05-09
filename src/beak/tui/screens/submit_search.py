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
from textual.widgets import Button, Input, Label, Select, Static

from ...project import BeakProject


_SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


class SubmitSearchModal(ModalScreen[Optional[str]]):
    """Submit an MMseqs2 search for the project's target sequence.

    Dismisses with the job_id on success, or None if cancelled / failed.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    /* Two-line blurb under the preset Select. Slightly indented and
       given a left accent so it reads as derived metadata, not
       another editable field. `height: auto` lets the second line
       wrap onto a third without clipping at the modal's right edge
       (which is what bit the previous single-line `Label`). */
    SubmitSearchModal #preset-blurb {
        height: auto;
        margin-top: 0;
        margin-bottom: 1;
        padding: 0 1;
        border-left: tall #2E86AB;
        color: $text;
    }
    """

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
            # Inline two-line blurb under the Select: tagline +
            # ROC1/param chips so the user sees what each preset
            # actually does (and the literature-derived ROC1 estimate
            # it carries) without having to read the source. `Static`
            # rather than `Label` because the blurb is multi-line.
            yield Static(
                self._preset_blurb("default"),
                id="preset-blurb",
                classes="preset-blurb",
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

    def _preset_blurb(self, preset: str) -> str:
        """Two-line blurb under the preset Select.

        Line 1 — tagline (the intent in 4–5 words).
        Line 2 — ROC1 chip + the params that actually moved between
                 presets, in chip form (`-s 7.5 · 2× iter · ≤5000 hits …`).
                 Numbers reflect what `mgr.submit(...preset=...)` will
                 actually pass to MMseqs2; the chips and the wire
                 args can't drift apart.
        """
        from ...remote.search import MMseqsSearch
        cfg = MMseqsSearch.PRESETS.get(preset, {})
        if not cfg:
            return ""
        tagline = cfg.get("tagline") or cfg.get("description") or ""
        params = cfg.get("params") or {}
        roc1 = cfg.get("roc1")

        chips = []
        if roc1 is not None:
            chips.append(f"[bold]ROC1 ≈ {roc1:.2f}[/bold]")
        else:
            chips.append("[bold]ROC1 —[/bold]")
        if "s" in params:
            chips.append(f"[dim]-s {params['s']}[/dim]")
        n_iter = params.get("num_iterations", 1)
        if n_iter > 1:
            chips.append(f"[dim]{n_iter}× iter[/dim]")
        if "max_seqs" in params:
            chips.append(f"[dim]≤{params['max_seqs']} hits[/dim]")
        if "min_seq_id" in params:
            chips.append(
                f"[dim]≥{int(round(params['min_seq_id'] * 100))}% id[/dim]"
            )
        if "c" in params:
            chips.append(
                f"[dim]≥{int(round(params['c'] * 100))}% cov[/dim]"
            )
        sep = " [dim]·[/dim] "
        return f"{tagline}\n{sep.join(chips)}"

    def on_select_changed(self, event) -> None:
        if getattr(event.select, "id", None) != "preset-select":
            return
        try:
            self.query_one("#preset-blurb", Static).update(
                self._preset_blurb(str(event.value))
            )
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        # Live-validate the set name as the user types so the regex
        # constraint surfaces before they hit Submit.
        if event.input.id != "set-name":
            return
        candidate = event.value.strip()
        if not candidate:
            # Empty is allowed — `_submit` falls back to "default".
            self._set_status("")
            return
        if _SET_NAME_RE.match(candidate):
            self._set_status(
                f"[dim]Will create / reuse set [bold]{candidate}[/bold].[/dim]"
            )
        else:
            self._set_status(
                "[red]Set name must start with a letter or digit and use "
                "only letters / digits / `_` / `-`.[/red]"
            )

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

        mgr = None
        try:
            from ...remote.search import MMseqsSearch
            mgr = MMseqsSearch()
            # Always pass the preset — `default` now carries filters
            # (max_seqs, coverage, min_seq_id) so skipping it on the
            # default branch would silently drop them.
            kwargs = {"database": db, "job_name": job_name}
            if preset:
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
        finally:
            # Release the SSH socket — without this, every search-modal
            # submission leaks a Paramiko Transport FD that lives until
            # the TUI exits.
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass

        self.app.call_from_thread(self.dismiss, job_id)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
