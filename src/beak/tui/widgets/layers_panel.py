"""Reactive layers panel with per-row clickable pills.

Each layer (target, homologs, ...) is one row. The homologs row hosts two
inline pills:
    - search-pill:        offers the Search action when no job exists
    - homologs-status:    shows live job status; clicking opens a detail modal

Pills are unified — both are `Pill(Label)` and post `Pill.Pressed(pill_id, value)`
on click. The parent screen routes by `pill_id`. State (which pill is visible,
its color/text) is recomputed by `refresh_state()`.

Status sources, in priority order:
    1. Cached remote-poll result (last `_poll_remote_status` worker run)
    2. Local `~/.beak/jobs.json` snapshot
    3. "running" placeholder when the job is tagged but never polled
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Label

from ...project import BeakProject


# Pfam domains run automatically on first project view (cheap hmmscan), so
# they don't need a manual layer affordance. Same goes for structures —
# AlphaFold fetch is automatic via StructureView.
_LAYER_ORDER = ("target", "homologs", "embeddings", "structures", "experiments")
_POLL_SECONDS = 15.0
_JOBS_DB = Path.home() / ".beak" / "jobs.json"
_BEAK_BLUE = "#2E86AB"

_UNIPROT_DBS = {
    "uniref50", "uniref90", "uniref100",
    "uniprotkb", "swissprot", "trembl",
}

_STATUS_STYLES = {
    "SUBMITTED": "cyan",
    "QUEUED":    "cyan",
    "RUNNING":   "yellow",
    "COMPLETED": "green",
    "FAILED":    "red",
    "CANCELLED": "dim",
    "UNKNOWN":   "dim",
}


def _close_mgr(mgr) -> None:
    """Best-effort close on a remote-job manager's SSH connection.

    Workers that instantiate fresh `MMseqsSearch` / `ClustalAlign` /
    `ESMEmbeddings` / `MMseqsTaxonomy` objects each open a Paramiko
    Transport. Without an explicit close, the socket FD is held until
    the process exits — a 15s poll cadence exhausts macOS's 256-FD
    limit in about 4 minutes.

    The bare `except Exception` is intentionally narrow — it swallows
    transient close errors (already-closed sockets, broken pipes) but
    lets KeyboardInterrupt / SystemExit propagate so the user can
    still abort a hung close with Ctrl-C.
    """
    try:
        conn = getattr(mgr, "conn", None)
        if conn is not None:
            conn.close()
    except Exception:
        pass


def _scratch_cleanup(mgr, job_id: str) -> None:
    """rm -rf the remote job dir + drop the local jobs.json entry.

    `~/beak_jobs/<id>/` is treated as scratch — once the artifact has
    landed in a project's local directory, the remote copy is
    redundant. Best-effort: SSH errors are swallowed so a flaky
    connection doesn't roll back a successful pull.
    """
    try:
        mgr.cleanup(job_id, keep_results=False)
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


class Pill(Label):
    """Inline clickable text pill. Hides itself when not active.

    Uses underline as a visual affordance for clickability without
    drawing a heavier border or button chrome.
    """

    class Pressed(Message):
        def __init__(self, pill_id: str, value: str = "") -> None:
            super().__init__()
            self.pill_id = pill_id
            self.value = value

    # Marquee budget for the body text inside the pill caps. Anything
    # longer scrolls instead of overflowing the row's slot, so the
    # pill width is stable regardless of label length and a long
    # status string ("aligning · running · queued") doesn't push
    # neighbouring columns around.
    BODY_MAX_W = 14
    # Ticks at offset=0 before scrolling resumes. Same cadence as the
    # project-list ticker (0.15s tick → ~2.5s pause).
    DWELL_TICKS = 17

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._active: bool = False
        self._value: str = ""
        # Marquee state — full body text, color, and current scroll
        # offset. `_dwell` counts down at offset=0 so the start of the
        # text gets a beat to be readable before scrolling resumes.
        self._full_text: str = ""
        self._color: str = "yellow"
        self._offset: int = 0
        self._dwell: int = 0
        # Start invisible — `display = False` drops the widget out of
        # layout so it consumes zero cells until `show()` swaps it in.
        # Without this, hidden pills still claim their padding budget,
        # which would push the visible pill far to the right when most
        # of the row's pills aren't active.
        self.display = False

    def show(self, text: str, color: str = "yellow", value: str = "") -> None:
        self._active = True
        self._value = value
        # Reset marquee position whenever the underlying text changes,
        # so a fresh status starts at offset 0 with a full reading
        # pause rather than wherever the previous text was scrolled to.
        if text != self._full_text:
            self._full_text = text
            self._offset = 0
            self._dwell = self.DWELL_TICKS
        self._color = color
        self._repaint()
        self.display = True

    def _repaint(self) -> None:
        """Push the pill's current marquee frame to the underlying Label.

        Named `_repaint` (not `_render`) because Textual's Widget base
        class has its own `_render()` in the rendering pipeline; shadowing
        it returns None where a Visual is expected and crashes
        `get_content_width` on first paint.
        """
        bg = "#1F2A3A"
        full = self._full_text
        if len(full) > self.BODY_MAX_W:
            # Loop the text with a small gap so the wrap doesn't snap.
            loop = full + "   " + full
            body = loop[self._offset:self._offset + self.BODY_MAX_W]
        else:
            body = full
        c = self._color
        self.update(
            f"[{c}]([/{c}]"
            f"[bold {c} on {bg}] {body} [/bold {c} on {bg}]"
            f"[{c}])[/{c}]"
        )

    def tick(self) -> None:
        """Advance the marquee one cell. Called by the panel's interval.

        No-op for inactive pills and for pills whose body fits the
        budget — short labels like "Search" or "running" stay
        stationary; only the "aligning · running" / "embedding ·
        queued" combos scroll.
        """
        if not self._active or len(self._full_text) <= self.BODY_MAX_W:
            return
        if self._dwell > 0:
            self._dwell -= 1
            return
        loop_len = len(self._full_text) + 3  # +3 for the gap segment
        self._offset = (self._offset + 1) % loop_len
        if self._offset == 0:
            self._dwell = self.DWELL_TICKS
        self._repaint()

    def hide_pill(self) -> None:
        self._active = False
        self._value = ""
        self._full_text = ""
        self._offset = 0
        self._dwell = 0
        self.update("")
        self.display = False

    def on_click(self, event) -> None:
        if self._active:
            self.post_message(self.Pressed(self.id or "", self._value))
            event.stop()  # don't let the row click handler fire too


class LayerRow(Horizontal):
    """Clickable layer row — opens the layer-detail modal."""

    class Clicked(Message):
        def __init__(self, layer_name: str) -> None:
            super().__init__()
            self.layer_name = layer_name

    def __init__(self, layer_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layer_name = layer_name

    def on_click(self, event) -> None:
        # Pills inside the row call event.stop() — so we only fire
        # when the click hits the row's plain (non-pill) area.
        self.post_message(self.Clicked(self._layer_name))


class LayersPanel(Vertical):
    """Layers panel with inline per-row pills + auto-refreshing status."""

    DEFAULT_CSS = """
    LayersPanel {
        height: auto;
        min-height: 10;
    }
    LayersPanel .layer-row { height: 1; }
    LayersPanel .layer-icon { width: 3; }
    LayersPanel .layer-name { width: 14; }
    /* `overflow-x: hidden` keeps over-long status pills (e.g.
       "(submitting align…)") from bleeding into the count column —
       they get clipped at the slot boundary instead. */
    LayersPanel .pill-slot {
        width: 24;
        height: 1;
        overflow-x: hidden;
    }
    LayersPanel Pill { width: auto; padding: 0 1; }
    /* `padding-right: 2` keeps the count column from butting right up
       against the size column on rows where the count text fills the
       width budget — previously `18,834 seqs (aligned)` ran straight
       into the size string with no breathing room. */
    LayersPanel .layer-count {
        width: 24;
        padding-right: 2;
    }
    LayersPanel .layer-size { width: 12; }
    LayersPanel Pill { width: auto; padding-right: 2; }
    """

    def __init__(self, project: BeakProject, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._remote_statuses: Dict[str, str] = {}
        self._mgr_unavailable: bool = False
        # Per-kind pull gates. A single shared flag previously meant a
        # poll cycle that finished both `align` and `tax` would silently
        # drop one of them; now each kind reentry-guards itself.
        self._pulling_kinds: set = set()
        self._submitting_align: bool = False  # gate against double-submit
        self._submitting_embed: bool = False
        # Live (done, total) for the current UniProt taxonomy auto-build,
        # set by `_build_taxonomy`'s progress callback so the homologs
        # row's status pill can render a counter. None when no build is
        # running.
        self._tax_progress: Optional[tuple] = None

    def compose(self) -> ComposeResult:
        for layer in _LAYER_ORDER:
            with LayerRow(layer, classes="layer-row", id=f"row-{layer}"):
                yield Label("", classes="layer-icon", id=f"icon-{layer}")
                yield Label(layer, classes="layer-name")
                # Pills sit in a fixed-width slot so the count + size
                # columns line up vertically across every row, whether
                # or not that layer has any active pills.
                with Horizontal(classes="pill-slot"):
                    if layer == "homologs":
                        yield Pill(id="search-pill")
                        yield Pill(id="align-pill")
                        yield Pill(id="tax-pill")
                        yield Pill(id="diff-pill")
                        yield Pill(id="homologs-status-pill")
                    elif layer == "embeddings":
                        yield Pill(id="embed-pill")
                        yield Pill(id="embeddings-status-pill")
                    elif layer == "experiments":
                        yield Pill(id="import-pill")
                yield Label("", classes="layer-count", id=f"count-{layer}")
                yield Label("", classes="layer-size", id=f"size-{layer}")

    # Pill marquee tick. Same cadence as the project-list ticker so
    # the two animations feel like one system. Cheap: each tick
    # iterates the panel's <= 25 Pill widgets and most early-return
    # because their text fits the budget or they're inactive.
    _PILL_MARQUEE_TICK_SECONDS = 0.15

    def on_mount(self) -> None:
        self.border_title = "Layers"
        self.refresh_state()
        self.set_interval(_POLL_SECONDS, self._maybe_poll)
        self.set_interval(
            self._PILL_MARQUEE_TICK_SECONDS, self._tick_pill_marquees
        )

    def _tick_pill_marquees(self) -> None:
        """Advance every active long-text pill by one cell."""
        for pill in self.query(Pill):
            pill.tick()

    def refresh_state(self) -> None:
        """Re-render rows from manifest + cached remote status. Cheap; no I/O."""
        manifest = self._project.manifest()
        sizes = self._project.disk_usage_by_layer()

        for layer in _LAYER_ORDER:
            present = self._layer_populated(layer, manifest)
            size = sizes.get(layer, 0)
            icon = "[green]✓[/green]" if present else "[dim]○[/dim]"
            size_str = _human_size(size) if size else "[dim]--[/dim]"
            count_str = self._layer_count(layer, manifest)
            self.query_one(f"#icon-{layer}", Label).update(icon)
            self.query_one(f"#count-{layer}", Label).update(count_str)
            self.query_one(f"#size-{layer}", Label).update(size_str)

        self._refresh_homologs_action(manifest)
        self._refresh_embeddings_action(manifest)
        self._refresh_experiments_action(manifest)

        self.border_subtitle = (
            f" {_human_size(self._project.disk_usage())} on disk "
        )

    def _layer_count(self, layer: str, manifest: dict) -> str:
        """Count cell text.

        Homologs uses a ratio when aligned (`17,234/18,834 aln`) so
        the count + alignment state collapse into a single phrase.
        Earlier versions had a separate green-parenthesized
        `(aligned)` tag that visually mimicked the clickable pills
        even though it wasn't interactive — confusing affordance.
        """
        data = manifest.get(layer) or {}
        if layer == "homologs":
            active = self._project.active_set() or {}
            n = active.get("n_homologs")
            if not n:
                return ""
            sets = data.get("sets") or []
            n_sets = len(sets)
            n_aln = active.get("n_aligned") or 0
            if n_aln:
                # Aligned ratio reads naturally and matches the
                # project-list marquee phrasing.
                label = f"[dim]{n_aln:,}/{n:,} aln[/dim]"
            elif (active.get("remote") or {}).get("align_job_id"):
                label = f"[dim]{n:,} hits · [/dim][yellow]aligning…[/yellow]"
            else:
                label = f"[dim]{n:,} hits[/dim]"
            if n_sets > 1:
                label += f"  [dim]· {n_sets} sets[/dim]"
            return label
        if layer == "embeddings":
            # Show every model embedded for the *active* homolog set,
            # plus a count of how many additional sets have embeddings.
            # Per-set / per-model totals live in the layer-detail modal
            # so this row stays compact.
            active_set = self._project.active_set_name()
            active_models = self._project.embeddings_models_for_set(active_set)
            active_models = [
                e for e in active_models if e.get("n_embeddings")
            ]
            if not active_models:
                return ""
            # Most-recently-updated first (per `embeddings_models_for_set`),
            # so the user sees the freshest model first when models pile up.
            label_parts = []
            for entry in active_models[:2]:
                m = entry.get("model") or "?"
                short = m.split("/")[-1]  # strip HF org prefix if any
                label_parts.append(f"{entry['n_embeddings']:,} {short}")
            label = "[dim]" + "  ·  ".join(label_parts) + "[/dim]"
            if len(active_models) > 2:
                label += f"  [dim]+{len(active_models) - 2}[/dim]"
            n_other = sum(
                1 for s in self._project.embeddings_sets()
                if s.get("n_embeddings")
                and s.get("source_homologs_set") != active_set
            )
            if n_other:
                label += f"  [dim]· +{n_other} other set{'s' if n_other > 1 else ''}[/dim]"
            return label
        if layer == "structures":
            d = self._project.path / "structures"
            if d.exists():
                n = len(list(d.glob("*.cif")))
                return f"[dim]{n}[/dim]" if n else ""
            return ""
        if layer == "experiments":
            return f"[dim]{len(data)}[/dim]" if isinstance(data, list) and data else ""
        return ""

    def _layer_populated(self, layer: str, manifest: dict) -> bool:
        """Whether a layer has *real data*, not just a tag."""
        data = manifest.get(layer)
        if not data:
            return False
        if layer == "target":
            return bool(data.get("sequence_file") or data.get("length"))
        if layer == "homologs":
            # "Done" means the active set has both hits AND alignment.
            active = self._project.active_set() or {}
            return bool(active.get("n_homologs")) and bool(active.get("n_aligned"))
        if layer == "embeddings":
            # ✓ icon when the *active* homolog set has embeddings —
            # other sets' embeddings don't count for the row state.
            active_emb = self._project.active_embeddings_set() or {}
            return bool(active_emb.get("n_embeddings"))
        if layer == "domains":
            return bool(data.get("n_hits") or data.get("hits"))
        if layer == "structures":
            return bool(data.get("n_structures"))
        if layer == "experiments":
            return isinstance(data, list) and len(data) > 0
        return bool(data)

    def force_refresh(self) -> None:
        """Refresh state AND kick off a remote poll right now.

        Debounces against an in-flight poll: the worker is decorated
        `exclusive=True` so a second call would queue, and on flaky
        networks a queued backlog leaves the UI feeling unresponsive
        as it drains. Skip the queue and surface a hint when the user
        mashes `r` while a poll is already running.
        """
        self.refresh_state()
        if not self._pending_jobs():
            return
        # Any pull or status fetch from a previous trigger still in
        # progress means the next 15 s tick (or the running worker
        # itself) will surface fresh state — no need to enqueue.
        if self._pulling_kinds:
            try:
                self.notify(
                    "Refresh already in progress…", timeout=2,
                )
            except Exception:
                pass
            return
        self._poll_remote_status()

    def _pending_jobs(self) -> Dict[str, List[str]]:
        """{kind: [job_id, ...]} for tagged-but-unfulfilled remote jobs.

        Reads the *active* homolog set for search/align/tax — sub-jobs
        for non-active sets aren't polled until the user switches.
        Embedding jobs are polled across *every* set, so a job
        submitted for set A continues to be tracked while the user is
        on set B; otherwise the pull would never run for the inactive
        set and the job would silently sit on the remote forever.
        """
        manifest = self._project.manifest()
        pending: Dict[str, List[str]] = {}
        active = self._project.active_set() or {}
        h_remote = active.get("remote") or {}
        if not active.get("n_homologs"):
            jid = h_remote.get("search_job_id")
            if jid:
                pending.setdefault("search", []).append(jid)
        else:
            if not active.get("n_aligned"):
                jid = h_remote.get("align_job_id")
                if jid:
                    pending.setdefault("align", []).append(jid)
            tax = manifest.get("taxonomy") or {}
            t_remote = tax.get("remote") or {}
            if not tax.get("n_assigned"):
                jid = t_remote.get("job_id")
                if jid:
                    pending.setdefault("tax", []).append(jid)

        for s in self._project.embeddings_sets():
            if s.get("n_embeddings"):
                continue
            jid = (s.get("remote") or {}).get("job_id")
            if jid:
                pending.setdefault("embed", []).append(jid)
        return pending

    def _refresh_homologs_action(self, manifest: dict) -> None:
        search_pill = self.query_one("#search-pill", Pill)
        align_pill = self.query_one("#align-pill", Pill)
        tax_pill = self.query_one("#tax-pill", Pill)
        diff_pill = self.query_one("#diff-pill", Pill)
        status_pill = self.query_one("#homologs-status-pill", Pill)

        # All homolog state now lives in the active set, not flat under
        # `[homologs]`. Sets are introduced lazily on first search.
        active = self._project.active_set() or {}
        h_remote = active.get("remote") or {}
        n_homologs = active.get("n_homologs", 0)
        n_aligned = active.get("n_aligned", 0)
        search_jid = h_remote.get("search_job_id")
        align_jid = h_remote.get("align_job_id")

        tax = manifest.get("taxonomy") or {}
        t_remote = tax.get("remote") or {}
        tax_jid = t_remote.get("job_id")
        tax_done = bool(tax.get("n_assigned"))

        # Search not yet done → only Search pill matters.
        if not n_homologs:
            search_pill.hide_pill()
            align_pill.hide_pill()
            tax_pill.hide_pill()
            diff_pill.hide_pill()
            if search_jid:
                status = self._lookup_status(search_jid)
                color = _STATUS_STYLES.get(status, "dim")
                status_pill.show(status.lower(), color=color, value=search_jid)
            else:
                search_pill.show("Search", color=_BEAK_BLUE, value="search")
                status_pill.hide_pill()
            return

        # Search is done. Hide Search; offer Align and/or Tax depending
        # on which sub-jobs are pending. Status pill takes precedence
        # while either Align or Tax is in flight.
        search_pill.hide_pill()

        if align_jid and not n_aligned:
            align_pill.hide_pill()
            tax_pill.hide_pill()
            diff_pill.hide_pill()
            status = self._lookup_status(align_jid)
            color = _STATUS_STYLES.get(status, "dim")
            status_pill.show(
                f"aligning · {status.lower()}", color=color, value=align_jid
            )
            return
        if tax_jid and not tax_done:
            align_pill.hide_pill() if not n_aligned else None
            # Show tax status pill; Align pill hidden but other states
            # (n_aligned) keep their indicator via the icon column.
            status = self._lookup_status(tax_jid)
            color = _STATUS_STYLES.get(status, "dim")
            status_pill.show(
                f"taxonomy · {status.lower()}", color=color, value=tax_jid
            )
            if not n_aligned:
                align_pill.hide_pill()
            tax_pill.hide_pill()
            return

        # No remote job in flight — but the local UniProt-REST taxonomy
        # build may still be churning through batches. Show its progress
        # in the status pill while it runs; the regular action pills
        # take over once it finishes.
        if self._tax_progress is not None:
            done, total = self._tax_progress
            if total > 0:
                pct = int(min(100, max(0, done / total * 100)))
                status_pill.show(
                    f"tax · {done:,}/{total:,} · {pct}%",
                    color="cyan",
                    value="",
                )
            else:
                status_pill.show("tax · starting…", color="cyan", value="")
        else:
            status_pill.hide_pill()
        if not n_aligned:
            align_pill.show("Align", color=_BEAK_BLUE, value="align")
        else:
            align_pill.hide_pill()

        # Only offer remote taxonomy when hits lack UniProt accessions —
        # for UniProt-based searches (uniref*, uniprotkb, etc.), the
        # auto UniProt-REST taxonomy already covers everything.
        search_db = (h_remote.get("search_database") or "").lower()
        needs_remote_tax = search_db and search_db not in _UNIPROT_DBS
        if needs_remote_tax and not tax_done:
            tax_pill.show("Tax", color=_BEAK_BLUE, value="tax")
        else:
            tax_pill.hide_pill()

        # Differential coloring needs a real alignment + a populated
        # traits.parquet (anything else and the modal would have nothing
        # to pick from). Cheap stat checks; no parsing.
        traits_path = self._project.active_homologs_dir() / "traits.parquet"
        if n_aligned and traits_path.exists():
            comp = manifest.get("comparative") or {}
            label = "Diff*" if comp.get("active_column") else "Diff"
            diff_pill.show(label, color=_BEAK_BLUE, value="diff")
        else:
            diff_pill.hide_pill()

    def _refresh_experiments_action(self, manifest: dict) -> None:
        try:
            pill = self.query_one("#import-pill", Pill)
        except Exception:
            return
        # Always offer import — multiple experiments per project are allowed.
        pill.show("Import", color=_BEAK_BLUE, value="import")

    def _refresh_embeddings_action(self, manifest: dict) -> None:
        embed_pill = self.query_one("#embed-pill", Pill)
        status_pill = self.query_one("#embeddings-status-pill", Pill)

        # Embeddings are now per-homolog-set: the row reflects whichever
        # set is currently active. Switching active sets shows that
        # set's embeddings if they exist, or the Embed action if they
        # don't — even when other sets already have embeddings.
        active_emb = self._project.active_embeddings_set() or {}
        remote = active_emb.get("remote") or {}
        n_embed = active_emb.get("n_embeddings", 0)
        n_homologs = (self._project.active_set() or {}).get("n_homologs", 0)

        # Stage A: this set's embeddings are ready.
        if n_embed:
            embed_pill.hide_pill()
            status_pill.hide_pill()
            return

        # Stage B: embedding job in flight for this set.
        job_id = remote.get("job_id")
        if job_id:
            embed_pill.hide_pill()
            status = self._lookup_status(job_id)
            color = _STATUS_STYLES.get(status, "dim")
            status_pill.show(
                f"embedding · {status.lower()}", color=color, value=job_id
            )
            return

        # Stage C: hits exist for this set, embeddings not started → offer Embed.
        if n_homologs:
            embed_pill.show("Embed", color=_BEAK_BLUE, value="embed")
            status_pill.hide_pill()
            return

        # Stage D: no hits yet → nothing to embed.
        embed_pill.hide_pill()
        status_pill.hide_pill()

    def _lookup_status(self, job_id: str) -> str:
        if job_id in self._remote_statuses:
            return self._remote_statuses[job_id]
        if _JOBS_DB.exists():
            try:
                with open(_JOBS_DB) as f:
                    db = json.load(f)
                if job_id in db:
                    return db[job_id].get("status", "UNKNOWN")
            except (json.JSONDecodeError, OSError):
                pass
        return "UNKNOWN"

    def _maybe_poll(self) -> None:
        if self._mgr_unavailable:
            return
        if self._pending_jobs():
            self._poll_remote_status()

    @work(thread=True, exclusive=True, group="status-poll")
    def _poll_remote_status(self) -> None:
        pending = self._pending_jobs()
        if not pending:
            return

        try:
            from ...remote.search import MMseqsSearch
            from ...remote.align import ClustalAlign
            from ...remote.embeddings import ESMEmbeddings
            from ...remote.taxonomy import MMseqsTaxonomy
        except Exception:
            self._mgr_unavailable = True
            return

        managers: Dict[str, object] = {}
        mgr_factories = {
            "search": MMseqsSearch,
            "align": ClustalAlign,
            "embed": ESMEmbeddings,
            "tax": MMseqsTaxonomy,
        }
        try:
            for kind, job_ids in pending.items():
                factory = mgr_factories.get(kind)
                if factory and kind not in managers:
                    try:
                        managers[kind] = factory()
                    except Exception:
                        self._mgr_unavailable = True
                        return

            for kind, job_ids in pending.items():
                mgr = managers.get(kind)
                if mgr is None:
                    continue
                for job_id in job_ids:
                    try:
                        info = mgr.status(job_id)
                        status = info.get("status", "UNKNOWN")
                        self._remote_statuses[job_id] = status
                        if status == "COMPLETED":
                            if kind == "search":
                                self._pull_homologs_now(mgr, job_id)
                            elif kind == "align":
                                self._pull_alignment_now(mgr, job_id)
                            elif kind == "embed":
                                self._pull_embeddings_now(mgr, job_id)
                            elif kind == "tax":
                                self._pull_taxonomy_now(mgr, job_id)
                    except Exception:
                        self._remote_statuses[job_id] = "UNKNOWN"
        finally:
            # Close every SSH connection we opened above. Without this
            # each 15s poll leaks a Paramiko Transport socket and macOS
            # hits its 256-FD soft limit within a few minutes (after
            # which even opening the project's manifest TOML fails with
            # OSError 24).
            for mgr in managers.values():
                _close_mgr(mgr)

        self.app.call_from_thread(self.refresh_state)

    @work(thread=True, exclusive=True, group="taxonomy-build")
    def _build_taxonomy(self) -> None:
        """Auto-build the taxonomy table after homologs land. Cheap once
        cached; the parquet is reused on subsequent project opens. For
        large hit sets this can take minutes — we publish per-batch
        progress to `_tax_progress` so the homologs row's status pill
        renders a live counter while we wait."""
        def _on_progress(done: int, total: int) -> None:
            self._tax_progress = (done, total)
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

        try:
            from ..taxonomy import build_taxonomy_table
            df = build_taxonomy_table(self._project, progress_cb=_on_progress)
            if df is None or df.empty:
                self._tax_progress = None
                self.app.call_from_thread(self.refresh_state)
                return
            n = len(df)
            n_resolved = int(df["organism"].notna().sum()) if "organism" in df else 0
            self.app.call_from_thread(
                self.notify, f"Taxonomy: {n_resolved}/{n} sequences", timeout=4
            )
            # Traits piggyback on taxonomy — cheap join once metaTraits is cached.
            self._build_traits()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Taxonomy build failed: {e}",
                severity="warning", timeout=6,
            )
        finally:
            self._tax_progress = None
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

    @work(thread=True, exclusive=True, group="traits-build")
    def _build_traits(self) -> None:
        """Join metaTraits onto the active set's taxonomy table.

        Silent when metaTraits isn't reachable — callers should treat
        traits as optional metadata, not a required layer.
        """
        try:
            from ..traits import build_traits_table
            df = build_traits_table(self._project)
            if df is None or df.empty:
                return
            n_matched = 0
            if "trait_match_level" in df.columns:
                n_matched = int(df["trait_match_level"].notna().sum())
            self.app.call_from_thread(
                self.notify,
                f"Traits: {n_matched}/{len(df)} matched",
                timeout=4,
            )
        except Exception:
            # Traits are optional — never let them surface a warning toast.
            pass

    def _pull_homologs_now(self, mgr, job_id: str) -> None:
        """Download hits.fasta into the project's homologs/ and tag the manifest."""
        if "homologs" in self._pulling_kinds:
            return
        self._pulling_kinds.add("homologs")
        try:
            self.app.call_from_thread(
                self.notify, f"Pulling homologs · {job_id}…"
            )
            result = mgr.get_results(job_id, parse=False, download_sequences=True)
            src_fasta = Path(result["fasta"])

            # Files land in the active set's directory.
            homologs_dir = self._project.active_homologs_dir(ensure=True)
            dest = homologs_dir / "sequences.fasta"
            import shutil
            shutil.copy2(src_fasta, dest)

            with open(dest) as f:
                n = sum(1 for line in f if line.startswith(">"))

            from datetime import datetime
            self._project.update_active_set(
                source="mmseqs",
                n_homologs=n,
                last_updated=datetime.now(),
            )

            self.app.call_from_thread(
                self.notify, f"Homologs ready · {n} sequences", timeout=8
            )
            # Local artifact landed → wipe the remote scratch dir.
            _scratch_cleanup(mgr, job_id)
            # Kick off the taxonomy fetch in parallel — it's UniProt
            # REST only, no SSH, takes seconds for hundreds of hits.
            self._build_taxonomy()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Pull failed: {e}",
                severity="error",
                timeout=10,
            )
        finally:
            self._pulling_kinds.discard("homologs")

    def _pull_alignment_now(self, mgr, job_id: str) -> None:
        """Download alignment.fasta into the active set + tag the manifest."""
        if "alignment" in self._pulling_kinds:
            return
        self._pulling_kinds.add("alignment")
        try:
            self.app.call_from_thread(
                self.notify, f"Pulling alignment · {job_id}…"
            )
            result = mgr.get_results(job_id)
            src = Path(result) if not isinstance(result, dict) else Path(
                result.get("alignment") or result.get("output") or ""
            )
            if not src or not src.exists():
                raise RuntimeError("Alignment file not produced")

            homologs_dir = self._project.active_homologs_dir(ensure=True)
            dest = homologs_dir / "alignment.fasta"
            import shutil
            shutil.copy2(src, dest)

            with open(dest) as f:
                n = sum(1 for line in f if line.startswith(">"))

            from datetime import datetime
            self._project.update_active_set(
                n_aligned=n, last_updated=datetime.now(),
            )

            cache = homologs_dir / "conservation.npy"
            if cache.exists():
                cache.unlink()

            self.app.call_from_thread(
                self.notify, f"Alignment ready · {n} sequences", timeout=8
            )
            _scratch_cleanup(mgr, job_id)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Pull failed: {e}",
                severity="error",
                timeout=10,
            )
        finally:
            self._pulling_kinds.discard("alignment")

    @work(thread=True, exclusive=True, group="align-submit")
    def _submit_alignment(self, set_name: Optional[str] = None) -> None:
        """Submit a Clustal Omega alignment for the named (or active) set."""
        target_set = set_name or self._project.active_set_name()
        hits_fasta = self._project.homologs_set_dir(target_set) / "sequences.fasta"
        if not hits_fasta.exists():
            self._submitting_align = False
            return
        mgr = None
        try:
            from ...remote.align import ClustalAlign
            mgr = ClustalAlign()
            job_name = f"{self._project.name}_{target_set}_align"
            job_id = mgr.submit(str(hits_fasta), job_name=job_name)

            # Stamp the named set's remote dict with the new job id —
            # not just the active one, so re-aligning a non-active set
            # from the modal records the job against the right entry.
            with self._project.mutate() as m:
                homologs = m.setdefault("homologs", {})
                for s in homologs.get("sets") or []:
                    if s.get("name") == target_set:
                        remote = dict(s.get("remote") or {})
                        remote["align_job_id"] = job_id
                        s["remote"] = remote
                        # Stale align artefact bookkeeping — re-align
                        # invalidates whatever counted as "aligned" before.
                        s.pop("n_aligned", None)
                        break

            self.app.call_from_thread(self.refresh_state)
            self.app.call_from_thread(
                self.notify,
                f"Alignment submitted for '{target_set}' · {job_id}",
                timeout=6,
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Align submit failed: {e}",
                severity="error",
                timeout=8,
            )
        finally:
            _close_mgr(mgr)
            # Clear the gate AFTER the worker thread has actually returned —
            # done synchronously here so a follow-up click is allowed once
            # the previous submission has finished (succeeded or failed).
            self._submitting_align = False
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

    def submit_alignment(self, set_name: Optional[str] = None) -> None:
        """Public entry point — kicks off the submit worker for `set_name`
        (defaults to the active set).

        Re-entrancy is gated synchronously: the flag flips on the UI
        thread before the worker spawns, so a double-click can't sneak
        a second `mgr.submit()` through while the first is in flight.
        """
        if self._submitting_align:
            return
        self._submitting_align = True

        try:
            self.query_one("#align-pill", Pill).hide_pill()
            self.query_one("#homologs-status-pill", Pill).show(
                "submitting…", color="cyan",
            )
        except Exception:
            pass

        self._submit_alignment(set_name)

    def _pull_embeddings_now(self, mgr, job_id: str) -> None:
        """Download embeddings tarball, extract into the per-set dir.

        We resolve the destination from the embeddings entry that owns
        this job_id rather than from `active_embeddings_dir()` — by the
        time the poll fires the job, the user may have switched active
        sets, but the tarball still belongs to the set it was submitted
        for. Same job-keyed lookup is used to update the manifest so
        we never write counts into the wrong entry.
        """
        # `_pulling_kinds` is a coarse global lock; key by job so two
        # different sets' pulls can run sequentially without one
        # blocking the other forever.
        key = f"embeddings:{job_id}"
        if key in self._pulling_kinds:
            return
        self._pulling_kinds.add(key)
        try:
            self.app.call_from_thread(
                self.notify, f"Pulling embeddings · {job_id}…"
            )
            embeddings_dir = self._project.embeddings_dir_for_job(job_id)
            if embeddings_dir is None:
                # Manifest no longer claims this job — e.g., the user
                # cleared it from another window. Skip silently rather
                # than rebuilding orphaned state.
                return
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            extracted = mgr.download(job_id, local_dir=str(embeddings_dir))
            n = sum(1 for _ in Path(extracted).rglob("*") if _.is_file())

            try:
                with open(_JOBS_DB) as f:
                    jdb = json.load(f)
                model = (jdb.get(job_id) or {}).get("model", "")
            except Exception:
                model = ""

            from datetime import datetime
            self._project.update_embeddings_set_by_job(
                job_id,
                n_embeddings=n,
                model=model,
                last_updated=datetime.now(),
            )

            self.app.call_from_thread(
                self.notify, f"Embeddings ready · {n} files", timeout=8
            )
            _scratch_cleanup(mgr, job_id)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Embeddings pull failed: {e}",
                severity="error", timeout=10,
            )
        finally:
            self._pulling_kinds.discard(key)

    @work(thread=True, exclusive=True, group="embed-submit")
    def _submit_embeddings(self) -> None:
        """Submit ESM2 embedding job over the active set's hits.fasta.

        This is the quick-action path (no modal); it always uses the
        ESMEmbeddings default model. For multi-model selection, the
        `SubmitEmbedModal` is the discoverable affordance.
        """
        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self._submitting_embed = False
            return
        mgr = None
        try:
            from ...remote.embeddings import ESMEmbeddings
            # Read the default off the manager class so this stays in
            # sync if the default ever changes.
            import inspect
            default_model = (
                inspect.signature(ESMEmbeddings.submit)
                .parameters["model"].default
            )
            mgr = ESMEmbeddings()
            job_name = f"{self._project.name}_embed"
            job_id = mgr.submit(str(hits_fasta), job_name=job_name)

            # Stamp the (active set, default model) entry with this
            # job. Creates a fresh entry if no embedding for this
            # (set, model) pair existed yet.
            self._project.update_active_embeddings_set(
                model=default_model,
                remote={"job_id": job_id},
            )

            self.app.call_from_thread(self.refresh_state)
            self.app.call_from_thread(
                self.notify, f"Embeddings submitted · {job_id}", timeout=6
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Embed submit failed: {e}",
                severity="error", timeout=8,
            )
        finally:
            _close_mgr(mgr)
            self._submitting_embed = False
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

    def _pull_taxonomy_now(self, mgr, job_id: str) -> None:
        """Pull MMseqs LCA results, save TSV + merge into taxonomy.parquet."""
        if "taxonomy" in self._pulling_kinds:
            return
        self._pulling_kinds.add("taxonomy")
        try:
            self.app.call_from_thread(
                self.notify, f"Pulling taxonomy · {job_id}…"
            )
            df = mgr.get_results(job_id, parse=True, parse_lineage=True)
            if df is None or len(df) == 0:
                raise RuntimeError("Empty taxonomy results")

            homologs_dir = self._project.active_homologs_dir(ensure=True)
            df.to_parquet(homologs_dir / "taxonomy_mmseqs.parquet", index=False)

            from datetime import datetime
            with self._project.mutate() as manifest:
                tax = manifest.setdefault("taxonomy", {})
                tax["n_assigned"] = int(len(df))
                tax["last_updated"] = datetime.now()

            self.app.call_from_thread(
                self.notify, f"Taxonomy ready · {len(df)} assignments", timeout=8
            )
            _scratch_cleanup(mgr, job_id)
            # MMseqs LCA may surface organisms the UniProt path missed —
            # rebuild traits to pick them up. Force=True via a stale cache:
            # _pull_taxonomy_now writes taxonomy_mmseqs.parquet, not the
            # canonical taxonomy.parquet, so the traits builder is a no-op
            # here unless the UniProt path also ran. Still cheap to call.
            self._build_traits()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Taxonomy pull failed: {e}",
                severity="error", timeout=10,
            )
        finally:
            self._pulling_kinds.discard("taxonomy")

    def submit_embeddings(self) -> None:
        """Public entry point — gated against double-submit."""
        if self._submitting_embed:
            return
        self._submitting_embed = True
        try:
            self.query_one("#embed-pill", Pill).hide_pill()
            self.query_one("#embeddings-status-pill", Pill).show(
                "submitting…", color="cyan",
            )
        except Exception:
            pass
        self._submit_embeddings()
