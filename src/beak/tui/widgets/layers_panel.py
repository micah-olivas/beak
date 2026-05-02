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
from typing import Dict

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

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._active: bool = False
        self._value: str = ""

    def show(self, text: str, color: str = "yellow", value: str = "") -> None:
        self._active = True
        self._value = value
        # Pill-style render: bold colored text on a subtle dark fill,
        # framed by parenthesis caps. Reads as a small button without
        # needing a per-line border.
        bg = "#1F2A3A"
        self.update(
            f"[{color}]([/{color}]"
            f"[bold {color} on {bg}] {text} [/bold {color} on {bg}]"
            f"[{color}])[/{color}]"
        )

    def hide_pill(self) -> None:
        self._active = False
        self._value = ""
        self.update("")

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
    LayersPanel .layer-count { width: 22; }
    LayersPanel .layer-size { width: 12; }
    LayersPanel Pill { width: auto; padding-right: 2; }
    """

    def __init__(self, project: BeakProject, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._remote_statuses: Dict[str, str] = {}
        self._mgr_unavailable: bool = False
        self._pulling: bool = False
        self._submitting_align: bool = False  # gate against double-submit
        self._submitting_embed: bool = False

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

    def on_mount(self) -> None:
        self.border_title = "Layers"
        self.refresh_state()
        self.set_interval(_POLL_SECONDS, self._maybe_poll)

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
        """Count cell text. Homologs shows hit count + '(aligned)' tag."""
        data = manifest.get(layer) or {}
        if layer == "homologs":
            active = self._project.active_set() or {}
            n = active.get("n_homologs")
            if not n:
                return ""
            sets = data.get("sets") or []
            n_sets = len(sets)
            label = f"[dim]{n:,} seqs[/dim]"
            if active.get("n_aligned"):
                label += "  [green](aligned)[/green]"
            elif (active.get("remote") or {}).get("align_job_id"):
                label += "  [yellow](aligning)[/yellow]"
            if n_sets > 1:
                label += f"  [dim]· {n_sets} sets[/dim]"
            return label
        if layer == "embeddings":
            n = data.get("n_embeddings")
            if not n:
                return ""
            return f"[dim]{n:,} vecs[/dim]"
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
            return bool(data.get("n_embeddings"))
        if layer == "domains":
            return bool(data.get("n_hits") or data.get("hits"))
        if layer == "structures":
            return bool(data.get("n_structures"))
        if layer == "experiments":
            return isinstance(data, list) and len(data) > 0
        return bool(data)

    def force_refresh(self) -> None:
        """Refresh state AND kick off a remote poll right now."""
        self.refresh_state()
        if self._pending_jobs():
            self._poll_remote_status()

    def _pending_jobs(self) -> Dict[str, str]:
        """{kind: job_id} for tagged-but-unfulfilled remote jobs.

        Reads the *active* homolog set; sub-jobs (align, tax) for
        non-active sets aren't polled until the user switches.
        """
        manifest = self._project.manifest()
        pending: Dict[str, str] = {}
        active = self._project.active_set() or {}
        h_remote = active.get("remote") or {}
        if not active.get("n_homologs"):
            jid = h_remote.get("search_job_id")
            if jid:
                pending["search"] = jid
        else:
            if not active.get("n_aligned"):
                jid = h_remote.get("align_job_id")
                if jid:
                    pending["align"] = jid
            tax = manifest.get("taxonomy") or {}
            t_remote = tax.get("remote") or {}
            if not tax.get("n_assigned"):
                jid = t_remote.get("job_id")
                if jid:
                    pending["tax"] = jid

        embeddings = manifest.get("embeddings") or {}
        e_remote = embeddings.get("remote") or {}
        if not embeddings.get("n_embeddings"):
            jid = e_remote.get("job_id")
            if jid:
                pending["embed"] = jid
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

        # No remote job in flight — offer the next available actions.
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

        embeddings = manifest.get("embeddings") or {}
        remote = embeddings.get("remote") or {}
        n_embed = embeddings.get("n_embeddings", 0)
        homologs = manifest.get("homologs") or {}
        n_homologs = (
            self._project.active_set() or {}
        ).get("n_homologs", 0) or homologs.get("n_homologs", 0)

        # Stage A: embeddings ready. If they were computed against a
        # different homolog set than the one currently active, surface
        # a "stale" pill so the user can rebuild instead of silently
        # mixing data.
        if n_embed:
            embed_pill.hide_pill()
            source = embeddings.get("source_homologs_set")
            active = self._project.active_set_name()
            if source and source != active:
                status_pill.show(
                    f"stale · from set {source}",
                    color="yellow",
                    value="embed-stale",
                )
            else:
                status_pill.hide_pill()
            return

        # Stage B: embedding job in flight.
        job_id = remote.get("job_id")
        if job_id:
            embed_pill.hide_pill()
            status = self._lookup_status(job_id)
            color = _STATUS_STYLES.get(status, "dim")
            status_pill.show(
                f"embedding · {status.lower()}", color=color, value=job_id
            )
            return

        # Stage C: hits exist, embeddings not started → offer Embed.
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

        managers = {}
        mgr_factories = {
            "search": MMseqsSearch,
            "align": ClustalAlign,
            "embed": ESMEmbeddings,
            "tax": MMseqsTaxonomy,
        }
        for kind, job_id in pending.items():
            factory = mgr_factories.get(kind)
            if factory and kind not in managers:
                try:
                    managers[kind] = factory()
                except Exception:
                    self._mgr_unavailable = True
                    return

        for kind, job_id in pending.items():
            mgr = managers.get(kind)
            if mgr is None:
                continue
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

        self.app.call_from_thread(self.refresh_state)

    @work(thread=True, exclusive=True, group="taxonomy-build")
    def _build_taxonomy(self) -> None:
        """Auto-build the taxonomy table after homologs land. Cheap once
        cached; the parquet is reused on subsequent project opens."""
        try:
            from ..taxonomy import build_taxonomy_table
            df = build_taxonomy_table(self._project)
            if df is None or df.empty:
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
        if self._pulling:
            return
        self._pulling = True
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
            self._pulling = False

    def _pull_alignment_now(self, mgr, job_id: str) -> None:
        """Download alignment.fasta into the active set + tag the manifest."""
        if self._pulling:
            return
        self._pulling = True
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
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Pull failed: {e}",
                severity="error",
                timeout=10,
            )
        finally:
            self._pulling = False

    @work(thread=True, exclusive=True, group="align-submit")
    def _submit_alignment(self) -> None:
        """Submit a Clustal Omega alignment of the active set's hits."""
        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self._submitting_align = False
            return
        try:
            from ...remote.align import ClustalAlign
            mgr = ClustalAlign()
            job_name = f"{self._project.name}_{self._project.active_set_name()}_align"
            job_id = mgr.submit(str(hits_fasta), job_name=job_name)

            # Stamp the active set's remote dict with the new job id.
            active = self._project.active_set() or {}
            remote = dict(active.get("remote") or {})
            remote["align_job_id"] = job_id
            self._project.update_active_set(remote=remote)

            self.app.call_from_thread(self.refresh_state)
            self.app.call_from_thread(
                self.notify, f"Alignment submitted · {job_id}", timeout=6
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"Align submit failed: {e}",
                severity="error",
                timeout=8,
            )
        finally:
            # Clear the gate AFTER the worker thread has actually returned —
            # done synchronously here so a follow-up click is allowed once
            # the previous submission has finished (succeeded or failed).
            self._submitting_align = False
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

    def submit_alignment(self) -> None:
        """Public entry point — kicks off the submit worker.

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

        self._submit_alignment()

    def _pull_embeddings_now(self, mgr, job_id: str) -> None:
        """Download embeddings tarball, extract into project's embeddings/."""
        if self._pulling:
            return
        self._pulling = True
        try:
            self.app.call_from_thread(
                self.notify, f"Pulling embeddings · {job_id}…"
            )
            embeddings_dir = self._project.path / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            extracted = mgr.download(job_id, local_dir=str(embeddings_dir))
            # Count vector files in the extracted dir.
            n = sum(1 for _ in Path(extracted).rglob("*") if _.is_file())

            # Pull the model name from jobs.json since download() doesn't
            # bring it back directly.
            try:
                with open(_JOBS_DB) as f:
                    jdb = json.load(f)
                model = (jdb.get(job_id) or {}).get("model", "")
            except Exception:
                model = ""

            from datetime import datetime
            manifest = self._project.manifest()
            emb = manifest.setdefault("embeddings", {})
            emb["n_embeddings"] = n
            emb["model"] = model
            emb["last_updated"] = datetime.now()
            # Stamp the source set on completion too — covers jobs that
            # were submitted before this tag was introduced.
            emb.setdefault(
                "source_homologs_set", self._project.active_set_name()
            )
            self._project.write(manifest)

            self.app.call_from_thread(
                self.notify, f"Embeddings ready · {n} files", timeout=8
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify, f"Embeddings pull failed: {e}",
                severity="error", timeout=10,
            )
        finally:
            self._pulling = False

    @work(thread=True, exclusive=True, group="embed-submit")
    def _submit_embeddings(self) -> None:
        """Submit ESM2 embedding job over the active set's hits.fasta."""
        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self._submitting_embed = False
            return
        try:
            from ...remote.embeddings import ESMEmbeddings
            mgr = ESMEmbeddings()
            job_name = f"{self._project.name}_embed"
            job_id = mgr.submit(str(hits_fasta), job_name=job_name)

            manifest = self._project.manifest()
            emb = manifest.setdefault("embeddings", {})
            # Tag embeddings with the homolog set they were computed
            # against so we can warn when the user switches active sets
            # without rebuilding.
            emb["source_homologs_set"] = self._project.active_set_name()
            remote = emb.setdefault("remote", {})
            remote["job_id"] = job_id
            self._project.write(manifest)

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
            self._submitting_embed = False
            try:
                self.app.call_from_thread(self.refresh_state)
            except Exception:
                pass

    def _pull_taxonomy_now(self, mgr, job_id: str) -> None:
        """Pull MMseqs LCA results, save TSV + merge into taxonomy.parquet."""
        if self._pulling:
            return
        self._pulling = True
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
            manifest = self._project.manifest()
            tax = manifest.setdefault("taxonomy", {})
            tax["n_assigned"] = int(len(df))
            tax["last_updated"] = datetime.now()
            self._project.write(manifest)

            self.app.call_from_thread(
                self.notify, f"Taxonomy ready · {len(df)} assignments", timeout=8
            )
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
            self._pulling = False

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
