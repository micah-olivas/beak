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

    def __init__(self, job_id: str, project=None) -> None:
        super().__init__()
        self._job_id = job_id
        # The project is used to clear the failed/cancelled job's ID
        # out of the manifest so the layers panel falls back to
        # offering the original action pill (Embed / Search / Align /
        # Tax). None when the modal is opened from a context that
        # isn't tied to a project (e.g., future `beak jobs` view).
        self._project = project
        # Two-click confirm: first click on "Cancel job" / "Clear"
        # arms the button; second click within the same modal session
        # actually performs the action. Prevents accidental cancels of
        # expensive long-running search/embedding jobs and accidental
        # clears of a job the user wanted to inspect further.
        self._cancel_armed = False
        self._cancel_in_flight = False
        self._clear_armed = False
        self._clear_in_flight = False
        self._last_status: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Job · {self._job_id}[/bold]", id="modal-title",
            )
            yield Static("[dim]Loading status…[/dim]", id="status-content")
            with Horizontal(id="modal-buttons"):
                cancel_btn = Button(
                    "Cancel job", id="cancel-btn", variant="warning"
                )
                # Hidden until the first poll confirms the job is in a
                # cancellable state — avoids inviting cancels on jobs
                # that already finished.
                cancel_btn.display = False
                yield cancel_btn
                clear_btn = Button(
                    "Clear", id="clear-btn", variant="error"
                )
                # Same pattern — hidden until we know the job is in a
                # terminal failure state worth clearing.
                clear_btn.display = False
                yield clear_btn
                yield Button("Close", id="close-btn")

    def on_mount(self) -> None:
        self._poll()
        # Refresh while the modal is open — running jobs progress.
        self.set_interval(5.0, self._poll)

    @work(thread=True, exclusive=True, group="job-status-detail")
    def _poll(self) -> None:
        mgr = None
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
                f"[dim]The remote job itself may still be running, or "
                f"may have crashed and the server is unreachable.\n"
                f"Use [bold]Clear[/bold] to drop this job from the "
                f"project locally — manifest cleanup runs without SSH "
                f"and will unstick the layers panel.[/dim]",
            )
            # Surface the Clear button even though we couldn't confirm
            # the remote state. The clear path's manifest cleanup is
            # local-only, so the user can always escape a stuck pill;
            # remote `cleanup()` will fail-soft inside `_do_clear` if
            # SSH stays broken. Treat the unknowable state as UNKNOWN
            # so `_update_clear_button` enables the button.
            self.app.call_from_thread(self._on_status_unreachable)
            return
        finally:
            # Drop the SSH socket between polls — the modal refreshes
            # every 5s, so without this each open modal leaks an FD per
            # tick and a long-running session exhausts the limit.
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass
        self.app.call_from_thread(self._render_status, info)

    def _render_status(self, info: dict) -> None:
        status = info.get("status", "UNKNOWN")
        runtime = info.get("runtime", "?")
        self._last_status = status
        self._update_cancel_button(status)
        self._update_clear_button(status)
        stages = info.get("stages") or []
        last_log = info.get("last_log_line") or ""
        # The base manager flattens `**progress` into the top-level
        # info dict, so we read the numeric counters directly instead
        # of from a nested "progress" key.
        current_op = info.get("current_operation")
        prefilter_step = info.get("prefilter_step")
        prefilter_total = info.get("total_prefilter_steps")
        align_step = info.get("align_step")
        align_total = info.get("total_align_steps")
        color = _STATUS_COLOR.get(status, "dim")

        lines = [
            f"[{color}]●[/{color}] [bold]{status}[/bold]   "
            f"[dim]runtime[/dim] {runtime}",
        ]

        n_sequences = info.get("n_sequences")
        if n_sequences:
            lines.append(f"[dim]Sequences:[/dim] {n_sequences:,}")

        if current_op:
            lines.append(f"[dim]Stage:[/dim] {current_op}")

        # Inline progress bars when MMseqs2 emits step counters.
        if prefilter_step and prefilter_total:
            lines.append(self._progress_bar(
                "Prefilter", prefilter_step, prefilter_total
            ))
        if align_step and align_total:
            lines.append(self._progress_bar(
                "Align", align_step, align_total
            ))

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

    @staticmethod
    def _progress_bar(label: str, current: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return f"[dim]{label}:[/dim] {current}"
        pct = current / total
        filled = max(0, min(width, int(round(pct * width))))
        bar = "█" * filled + "░" * (width - filled)
        return (
            f"[dim]{label}:[/dim] [#65CBF3]{bar}[/#65CBF3] "
            f"{current}/{total} [dim]({pct * 100:.0f}%)[/dim]"
        )

    def _set_content(self, text: str) -> None:
        # Guard against dismissal-while-poll-in-flight: the worker's
        # `app.call_from_thread` can land here after the modal was
        # closed, in which case the widget is gone and `query_one`
        # raises NoMatches. Silently no-op rather than crash.
        try:
            self.query_one("#status-content", Static).update(text)
        except Exception:
            pass

    def _update_cancel_button(self, status: str) -> None:
        """Show + label the cancel button based on current job status.

        Only RUNNING / SUBMITTED jobs are cancellable; for everything
        else we hide the button so the modal stays a clean read-only
        view. The cancel-armed state is reset whenever the job leaves
        a cancellable status (e.g., it finished on its own)."""
        try:
            btn = self.query_one("#cancel-btn", Button)
        except Exception:
            return  # modal still composing
        cancellable = status in ("RUNNING", "SUBMITTED")
        if self._cancel_in_flight:
            btn.display = True
            btn.disabled = True
            btn.label = "Cancelling…"
            return
        btn.display = cancellable
        btn.disabled = not cancellable
        if not cancellable:
            self._cancel_armed = False
            btn.label = "Cancel job"
        elif self._cancel_armed:
            btn.label = "Confirm cancel"
        else:
            btn.label = "Cancel job"

    def _on_status_unreachable(self) -> None:
        """Hand the user a Clear escape hatch when the status poll itself
        threw (SSH down, server gone, jobs.json corrupt). We can't
        verify the remote, but the local manifest can still be cleaned
        up, which is what unsticks the layers panel."""
        self._last_status = "UNKNOWN"
        # Hide Cancel — we can't kill what we can't reach.
        try:
            cancel_btn = self.query_one("#cancel-btn", Button)
            cancel_btn.display = False
        except Exception:
            pass
        self._update_clear_button("UNKNOWN")

    def _update_clear_button(self, status: str) -> None:
        """Show + label the clear button for terminal failure states.

        Clearing is offered for FAILED / CANCELLED / UNKNOWN — anything
        that left a job_id pinned in the manifest with no live work to
        match it. We don't offer it for COMPLETED (the manifest is
        already updated to point at real outputs), or for RUNNING /
        SUBMITTED (cancel first, then clear)."""
        try:
            btn = self.query_one("#clear-btn", Button)
        except Exception:
            return
        clearable = status in ("FAILED", "CANCELLED", "UNKNOWN")
        if self._clear_in_flight:
            btn.display = True
            btn.disabled = True
            btn.label = "Clearing…"
            return
        btn.display = clearable
        btn.disabled = not clearable
        if not clearable:
            self._clear_armed = False
            btn.label = "Clear"
        elif self._clear_armed:
            btn.label = "Confirm clear"
        else:
            btn.label = "Clear"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss(None)
            return
        if event.button.id == "clear-btn":
            if self._clear_in_flight:
                return
            if not self._clear_armed:
                self._clear_armed = True
                self._update_clear_button(self._last_status or "FAILED")
                return
            self._clear_in_flight = True
            self._update_clear_button(self._last_status or "FAILED")
            self._set_content(
                "[yellow]●[/yellow] [bold]Clearing…[/bold]\n\n"
                "[dim]Removing the remote job directory and clearing "
                "this job ID from the project manifest. The action "
                "pill will come back once this is done.[/dim]"
            )
            self._do_clear()
            return
        if event.button.id == "cancel-btn":
            if self._cancel_in_flight:
                return
            if not self._cancel_armed:
                # First click: arm the button. The user has to click
                # again to actually send the cancel — defends against
                # fat-fingering on expensive jobs.
                self._cancel_armed = True
                self._update_cancel_button(self._last_status or "RUNNING")
                return
            # Second click: fire the cancel.
            self._cancel_in_flight = True
            self._update_cancel_button(self._last_status or "RUNNING")
            self._set_content(
                "[yellow]●[/yellow] [bold]Cancelling…[/bold]\n\n"
                "[dim]Sending SIGTERM to the job's process tree on "
                "the remote, then SIGKILL after 2s. This may take a "
                "few seconds over SSH.[/dim]"
            )
            self._do_cancel()

    @work(thread=True, exclusive=True, group="job-cancel")
    def _do_cancel(self) -> None:
        mgr = None
        err: str | None = None
        try:
            from ...cli._common import get_manager
            mgr = get_manager(job_id=self._job_id)
            mgr.cancel(self._job_id)
        except Exception as e:  # noqa: BLE001
            err = (str(e).splitlines()[0] if str(e) else type(e).__name__)
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass
        self.app.call_from_thread(self._after_cancel, err)

    def _after_cancel(self, err: str | None) -> None:
        self._cancel_in_flight = False
        if err:
            # Cancel failed — surface the reason but leave the button
            # available so the user can retry.
            self._cancel_armed = False
            self._update_cancel_button(self._last_status or "RUNNING")
            self._set_content(
                f"[red]Cancel failed:[/red] [dim]{err[:200]}[/dim]\n\n"
                "[dim]Try again, or kill the process manually on the "
                "remote.[/dim]"
            )
            return
        # Force an immediate refresh so the modal flips to CANCELLED
        # without waiting for the next 5s tick.
        self._cancel_armed = False
        self._poll()

    @work(thread=True, exclusive=True, group="job-clear")
    def _do_clear(self) -> None:
        mgr = None
        err: str | None = None
        job_type: str | None = None
        try:
            from ...cli._common import get_manager
            mgr = get_manager(job_id=self._job_id)
            # Capture the type before cleanup deletes the local entry.
            job_db = mgr._load_job_db()
            job_type = (job_db.get(self._job_id) or {}).get("job_type")
            # cleanup() removes the remote job directory and the entry
            # in ~/.beak/jobs.json. Pass keep_results=False since the
            # job failed — there's nothing useful in tmp/ either.
            mgr.cleanup(self._job_id, keep_results=False)
        except Exception as e:  # noqa: BLE001
            err = (str(e).splitlines()[0] if str(e) else type(e).__name__)
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass

        # Clear the job_id from the project manifest. Best-effort: even
        # if the remote cleanup above failed (e.g., SSH dropped), still
        # try to unstick the manifest so the user can move on. We use
        # job_type when we got it, otherwise scan all known slots.
        if self._project is not None:
            try:
                self._clear_manifest_job_id(job_type)
            except Exception as e:  # noqa: BLE001
                if not err:
                    err = (
                        f"manifest cleanup: "
                        f"{(str(e).splitlines()[0] if str(e) else type(e).__name__)}"
                    )

        self.app.call_from_thread(self._after_clear, err)

    def _clear_manifest_job_id(self, job_type: str | None) -> None:
        """Pop this job_id from whichever manifest slot owns it.

        The mapping mirrors how the layers panel reads job IDs:
            search    → homologs.sets[active].remote.search_job_id
            align     → homologs.sets[active].remote.align_job_id
            taxonomy  → taxonomy.remote.job_id
            embeddings→ embeddings.remote.job_id

        When job_type is None (the local DB entry was already gone)
        we scan every slot and pop any match — defensive against a
        partially-cleaned-up state."""
        target_id = self._job_id

        def _pop_if_match(remote: dict, key: str) -> bool:
            if remote.get(key) == target_id:
                remote.pop(key, None)
                return True
            return False

        with self._project.mutate() as m:
            check_search = job_type in (None, "search")
            check_align = job_type in (None, "align")
            check_tax = job_type in (None, "taxonomy", "tax")
            check_embed = job_type in (None, "embeddings", "embed")

            if check_search or check_align:
                sets = (m.get("homologs") or {}).get("sets") or []
                for s in sets:
                    rem = s.get("remote")
                    if not rem:
                        continue
                    if check_search:
                        _pop_if_match(rem, "search_job_id")
                    if check_align:
                        _pop_if_match(rem, "align_job_id")

            if check_tax:
                rem = (m.get("taxonomy") or {}).get("remote")
                if rem:
                    _pop_if_match(rem, "job_id")

            if check_embed:
                # Embeddings are now per-(homolog set, model): walk
                # `embeddings.sets`, pop whichever `remote.job_id`
                # matches, and drop entries that have nothing left
                # (no `n_embeddings`, no `remote`). For dropped
                # entries we *also* rmtree the per-(set, model) dir
                # — without that, a failed embed leaves orphan files
                # under `embeddings/<set>/<model_slug>/` that the
                # next embed for the same pair would overwrite into.
                # An entry with surviving `n_embeddings` from a prior
                # successful run is left intact (just the failed
                # job_id gets dropped), so re-submitting against an
                # already-good model doesn't lose its data.
                emb = m.get("embeddings") or {}
                sets = emb.get("sets") or []
                kept = []
                self._embed_dirs_to_rm: list[tuple[str, str | None]] = []
                for s in sets:
                    rem = s.get("remote") or {}
                    if rem.get("job_id") == target_id:
                        rem.pop("job_id", None)
                        if rem:
                            s["remote"] = rem
                        else:
                            s.pop("remote", None)
                    if s.get("n_embeddings") or s.get("remote"):
                        kept.append(s)
                    else:
                        set_name = s.get("source_homologs_set")
                        if set_name:
                            self._embed_dirs_to_rm.append(
                                (set_name, s.get("model"))
                            )
                if kept:
                    emb["sets"] = kept
                    m["embeddings"] = emb
                elif "embeddings" in m:
                    m.pop("embeddings", None)

        # Outside the manifest lock: rmtree any dirs flagged for the
        # embed branch above. Lock is per-project (project.mutate); we
        # don't want filesystem I/O blocking other manifest writers.
        dirs = getattr(self, "_embed_dirs_to_rm", None) or []
        if dirs:
            import shutil
            for set_name, model in dirs:
                d = self._project.embeddings_set_dir(set_name, model=model)
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            self._embed_dirs_to_rm = []

    def _after_clear(self, err: str | None) -> None:
        self._clear_in_flight = False
        if err:
            self._clear_armed = False
            self._update_clear_button(self._last_status or "FAILED")
            self._set_content(
                f"[red]Clear failed:[/red] [dim]{err[:200]}[/dim]\n\n"
                "[dim]The remote job directory may be partially "
                "cleaned. Try again, or run "
                f"`beak jobs cleanup {self._job_id}` from a shell.[/dim]"
            )
            return
        # Dismiss with a truthy value so the parent's callback knows
        # to refresh the layers panel. The pill will swap from
        # "embedding · failed" back to the "Embed" action.
        self.dismiss({"cleared": True})

    def action_close(self) -> None:
        self.dismiss(None)
