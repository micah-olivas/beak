"""InterPLM feature viewer.

Two-pane screen accessible via `i` from the project detail screen:

  Left  — DataTable of top SAE features ranked by max activation
  Right — Protein sequence colored by per-residue activation for the
          currently highlighted feature

The ESM-2-8M model and SAE weights are downloaded on first use and
cached at ~/.beak/interplm/. Requires: pip install torch transformers
huggingface_hub  (or `pip install "beak[interplm]"`).
"""

from __future__ import annotations

import colorsys
from typing import List, Optional, Tuple


import numpy as np
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Static
from textual.widget import Widget

from ...project import BeakProject
from ..widgets.server_status import ServerStatusBar


# ── Activation gradient stops (value 0→1, RGB tuples) ─────────────────────
_GRAD_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (0x2D, 0x31, 0x39)),  # near-black (inactive)
    (0.20, (0x1F, 0x4B, 0x7A)),  # dark blue
    (0.50, (0x65, 0xCB, 0xF3)),  # cyan
    (0.75, (0xFF, 0xA6, 0x2B)),  # amber
    (1.00, (0xFF, 0x40, 0x40)),  # red
]

_RESIDUES_PER_LINE = 50


def _act_color(v: float) -> str:
    """Map activation value 0–1 to a '#RRGGBB' background color."""
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(_GRAD_STOPS) - 1):
        lo_v, lo_c = _GRAD_STOPS[i]
        hi_v, hi_c = _GRAD_STOPS[i + 1]
        if v <= hi_v:
            t = (v - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.0
            r = int(lo_c[0] + t * (hi_c[0] - lo_c[0]))
            g = int(lo_c[1] + t * (hi_c[1] - lo_c[1]))
            b = int(lo_c[2] + t * (hi_c[2] - lo_c[2]))
            return f"#{r:02X}{g:02X}{b:02X}"
    return f"#{_GRAD_STOPS[-1][1][0]:02X}{_GRAD_STOPS[-1][1][1]:02X}{_GRAD_STOPS[-1][1][2]:02X}"


def _fg_for_bg(hex_bg: str) -> str:
    """Light or dark foreground text for readability on hex_bg."""
    r = int(hex_bg[1:3], 16)
    g = int(hex_bg[3:5], 16)
    b = int(hex_bg[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#FFFFFF" if luminance < 140 else "#111111"



class _SeqPane(Widget):
    """Renders the target sequence with per-residue activation background colors.

    `set_feature(sequence, acts)` triggers a refresh; until then shows a
    placeholder. Uses coalesced rich.text.Text appends (same pattern as
    alignment_view._coalesce_text) — never markup strings.
    """

    DEFAULT_CSS = """
    _SeqPane {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sequence: str = ""
        self._acts: Optional[np.ndarray] = None  # shape (L,)

    def set_feature(self, sequence: str, acts: np.ndarray) -> None:
        self._sequence = sequence
        self._acts = acts.astype(np.float32)
        self.refresh()

    def render(self) -> Text:
        if not self._sequence or self._acts is None:
            return Text("Select a feature to view activation pattern.", style="dim")

        seq = self._sequence
        acts = self._acts
        out = Text(no_wrap=False)

        for line_start in range(0, len(seq), _RESIDUES_PER_LINE):
            chunk = seq[line_start: line_start + _RESIDUES_PER_LINE]
            chunk_acts = acts[line_start: line_start + len(chunk)]

            # Position prefix
            pos_label = f"{line_start + 1:>5}  "
            out.append(pos_label, style="dim")

            # Coalesced color runs — each run covers consecutive residues
            # sharing the same background color bucket.
            run_bg: Optional[str] = None
            run_fg: Optional[str] = None
            run_start = 0

            for i, (aa, v) in enumerate(zip(chunk, chunk_acts)):
                bg = _act_color(v)
                fg = _fg_for_bg(bg)
                if bg != run_bg:
                    if i > run_start:
                        out.append(
                            chunk[run_start:i],
                            style=f"{run_fg} on {run_bg}",
                        )
                    run_bg = bg
                    run_fg = fg
                    run_start = i

            if run_start < len(chunk):
                out.append(
                    chunk[run_start:],
                    style=f"{run_fg} on {run_bg}",
                )

            out.append("\n")

        return out


class _ModelLayerPanel(Widget):
    """Visual model/layer selector for the InterPLM screen.

    One row per model, layer chips beside it.
    Chip styles: active = reverse, computed = amber #F5A623,
    running = yellow ⟳, not-computed = dim.
    Clicking a chip fires _ModelLayerPanel.Selected(model, layer).
    """

    class Selected(Message):
        def __init__(self, model: str, layer: int) -> None:
            super().__init__()
            self.model = model
            self.layer = layer

    DEFAULT_CSS = """
    _ModelLayerPanel {
        height: auto;
        margin-bottom: 1;
        border-bottom: solid #2E86AB;
    }
    """

    def __init__(self, model: str, layer: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_model = model
        self._active_layer = layer
        self._chip_map: list = []  # (row_y, x_start, x_end, model_key, layer)

    def on_mount(self) -> None:
        self.set_interval(1.5, self.refresh)

    def set_selection(self, model: str, layer: int) -> None:
        self._active_model = model
        self._active_layer = layer
        self.refresh()

    def render(self) -> Text:
        from ... import interplm

        self._chip_map = []
        jobs = interplm.get_active_concepts_jobs()
        running_jobs = {
            (m, l)
            for (m, l), rec in jobs.items()
            if rec.get("status") == "running"
        }

        out = Text(no_wrap=True)
        for row_y, (mk, cfg) in enumerate(interplm._MODELS.items()):
            active_model = mk == self._active_model
            short = cfg["label"].replace("ESM-2-", "")  # "8M" or "650M"
            prefix = f"  {short:<6} "
            out.append(prefix, style="bold #2E86AB" if active_model else "dim")

            x = len(prefix)
            for lyr in cfg["layers"]:
                active = active_model and lyr == self._active_layer
                running = (mk, lyr) in running_jobs
                computed = interplm.has_concepts(lyr, mk)

                if running:
                    chip = f"⟳{lyr}"
                    sty = "bold yellow"
                elif active:
                    chip = str(lyr)
                    sty = "bold reverse"
                elif computed:
                    chip = str(lyr)
                    sty = "#F5A623"
                else:
                    chip = str(lyr)
                    sty = "dim"

                self._chip_map.append((row_y, x, x + len(chip), mk, lyr))
                out.append(chip, style=sty)
                out.append("  ")
                x += len(chip) + 2

            out.append("\n")

        return out

    def on_click(self, event) -> None:
        for ry, x0, x1, mk, lyr in self._chip_map:
            if event.y == ry and x0 <= event.x < x1:
                self.post_message(self.Selected(mk, lyr))
                return


class InterPLMScreen(Screen):
    """Full-screen InterPLM SAE feature viewer."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("j", "next_feature", "Next"),
        Binding("k", "prev_feature", "Prev"),
        Binding("c", "compute_concepts", "Concepts"),
        Binding("s", "toggle_proteins", "Swiss-Prot"),
        Binding("p", "precompute_swissprot", "Precompute SW"),
    ]
    # Model / layer cycling lives entirely on the chip panel now —
    # `m` / `[` / `]` were redundant once the panel landed and only
    # cluttered the footer.

    DEFAULT_CSS = """
    InterPLMScreen #ipl-row {
        width: 100%;
        height: 1fr;
    }
    InterPLMScreen #feature-table-wrap {
        width: 42;
        height: 100%;
        border-right: solid #2E86AB;
    }
    InterPLMScreen #feature-table {
        width: 100%;
        height: 1fr;
    }
    InterPLMScreen #seq-col {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }
    InterPLMScreen #feature-label {
        height: auto;
        padding: 1 0 0 0;
        color: $text;
    }
    InterPLMScreen #seq-pane {
        height: 1fr;
        overflow-y: auto;
    }
    InterPLMScreen #proteins-table {
        height: 1fr;
        display: none;
    }
    InterPLMScreen #feature-stats {
        height: auto;
        padding: 0 0 1 0;
        color: $text-muted;
    }
    InterPLMScreen #loading-label {
        padding: 2 3;
        color: $text-muted;
    }
    InterPLMScreen #citation-label {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._user_closing: bool = False
        self._error: Optional[str] = None
        self._feature_acts: Optional[np.ndarray] = None  # (L, n_features)
        self._top_features: list = []
        self._sequence: str = ""
        self._model: str = "8m"
        self._layer: int = 6
        self._domains: list = []  # Pfam hits: [{pfam_name, pfam_id, env_from, env_to}]
        self._concepts: dict = {}  # feature_idx → concept name (from concepts CSV)
        self._right_mode: str = "sequence"  # "sequence" or "proteins"
        self._sw_top_ids: Optional[np.ndarray] = None   # (n_feat, 100) dtype U10
        self._sw_top_acts: Optional[np.ndarray] = None  # (n_feat, 100) dtype f16
        # (model, layer) of the concept job currently running, or None.
        self._concept_job: Optional[tuple] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ServerStatusBar(id="server-status")
        with Horizontal(id="ipl-row"):
            with Vertical(id="feature-table-wrap"):
                yield Label("[dim]Loading…[/dim]", id="loading-label")
                yield DataTable(id="feature-table", cursor_type="row")
            with Vertical(id="seq-col"):
                yield _ModelLayerPanel(self._model, self._layer, id="model-layer-panel")
                yield Static("", id="feature-label")
                yield _SeqPane(id="seq-pane")
                yield DataTable(id="proteins-table", cursor_type="row")
                yield Static("", id="feature-stats")
        yield Static(
            "[dim]Simon & Zou. InterPLM. [italic]Nature Methods[/italic]"
            " 22, 2107–2117 (2025). doi:10.1038/s41592-025-02836-7[/dim]",
            id="citation-label",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "InterPLM Features"
        self._update_subtitle()

        try:
            manifest = self._project.manifest()
            self._domains = list((manifest.get("domains") or {}).get("hits") or [])
        except Exception:
            self._domains = []

        try:
            from ... import interplm
            self._concepts = interplm.load_concepts(self._layer, self._model)
        except Exception:
            self._concepts = {}

        try:
            from ... import interplm
            if interplm.has_swissprot_top100(self._layer, self._model):
                self._sw_top_ids, self._sw_top_acts = interplm.load_swissprot_top100(
                    self._layer, self._model
                )
        except Exception:
            pass

        table = self.query_one("#feature-table", DataTable)
        table.add_columns("Rank", "Feature", "Concept", "Max")
        table.display = False

        pt = self.query_one("#proteins-table", DataTable)
        pt.add_columns("Rank", "Accession", "Max Act")
        pt.display = False

        # If a concept job started in a previous mount of this screen
        # (or in another beak ui session for this user) is still running,
        # adopt its state so the loading label, subtitle, and chip panel
        # reflect "computing" instead of looking idle. Without this the
        # user can pop back to the screen mid-job and have no signal that
        # work is in flight.
        from ... import interplm
        for (m, l), rec in interplm.get_active_concepts_jobs().items():
            if rec.get("status") == "running":
                self._concept_job = (m, l)
                self._update_subtitle()
                self._set_loading_label(rec.get("message") or "Computing concepts…")
                break

        # Sync local UI with global concept-job state at 2s cadence.
        # Cheap: each tick reads one dict + maybe refreshes one widget.
        self.set_interval(2.0, self._tick_concept_status)

        self._load_async()

    def _tick_concept_status(self) -> None:
        """Pull the running-job message from globals so the screen stays
        live even if the worker that started the job belonged to a
        previous mount. Also detects when a job finishes outside of
        ``_on_concept_finished`` and reloads concepts from disk.
        """
        from ... import interplm
        if self._user_closing:
            return
        active = interplm.get_active_concepts_jobs()
        running = next(
            ((m, l, rec) for (m, l), rec in active.items()
             if rec.get("status") == "running"),
            None,
        )
        if running is not None:
            m, l, rec = running
            if self._concept_job != (m, l):
                self._concept_job = (m, l)
                self._update_subtitle()
            msg = rec.get("message")
            if msg:
                self._set_loading_label(msg)
        elif self._concept_job is not None:
            # Job vanished from globals — completed (good), failed, or
            # expired past the 30s TTL. If the on-disk CSV is fresher
            # than what we have in memory, refresh the table so the new
            # labels show up without waiting for a manual reload.
            self._concept_job = None
            self._update_subtitle()
            try:
                fresh = interplm.load_concepts(self._layer, self._model)
                if fresh != self._concepts:
                    self._concepts = fresh
                    self._refresh_concepts_in_table(len(fresh))
            except Exception:
                pass

        # Chip panel reads global state directly — refresh so its
        # ⟳-running indicator stays in sync.
        try:
            self.query_one("#model-layer-panel", _ModelLayerPanel).refresh()
        except Exception:
            pass

    @work(thread=True, exclusive=True, group="interplm-load")
    def _load_async(self) -> None:
        from ... import interplm

        try:
            self._sequence = self._project.target_sequence()

            if interplm.is_cached(self._project.path, self._layer, self._model):
                self._progress("Loading cached features…")
                result = interplm.load_from_cache(
                    self._project.path, self._layer, self._model
                )
            else:
                result = interplm.compute_features(
                    self._sequence,
                    layer=self._layer,
                    model=self._model,
                    progress_cb=self._progress,
                )
                self._progress("Saving to cache…")
                interplm.save_to_cache(self._project.path, result)

            self._feature_acts = result["feature_acts"]
            self._top_features = result["top_features"]
        except ImportError:
            # transformers raises ImportError("EsmModel requires the PyTorch
            # library") when torch isn't installed, even if the ensure_deps
            # install succeeded but the module cache wasn't cleared in time.
            self._error = (
                "PyTorch not installed in this environment.\n\n"
                "Run:  conda install pytorch cpuonly -c pytorch\n"
                "  or: pip install torch\n\n"
                "Then restart beak."
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            from datetime import datetime
            from pathlib import Path
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path.home() / ".beak" / "errors"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{ts}_interplm.log"
                with open(log_path, "w") as f:
                    f.write(f"# interplm failure · {ts}\n")
                    f.write(f"# project: {self._project.name}\n\n")
                    f.write(f"{type(e).__name__}: {e}\n\n")
                    f.write("--- traceback ---\n")
                    traceback.print_exception(
                        type(e), e, e.__traceback__, file=f,
                    )
                self._error = f"{type(e).__name__}: {e}\n\nFull log: {log_path}"
            except Exception:
                self._error = f"{type(e).__name__}: {e}"

        try:
            loop = self.app._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._load_done)
        except Exception:  # noqa: BLE001
            pass

    def _progress(self, msg: str) -> None:
        """Called from the worker thread to update the loading label."""
        try:
            loop = self.app._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._set_loading_label, msg)
        except Exception:
            pass

    def _set_loading_label(self, msg: str) -> None:
        if self._user_closing:
            return
        try:
            lbl = self.query_one("#loading-label", Label)
            lbl.update(f"[dim]{msg}[/dim]")
            lbl.display = True
        except Exception:
            pass

    def _concept_for_feature(self, feat_idx: int) -> str:
        """Return computed concept or '—'.

        Earlier versions fell back to the residue's Pfam domain when no
        computed concept existed. That collapsed every feature to the
        protein's single Pfam name (e.g. "Acylphosphatase" for ACYP1),
        masked which features actually got SAE-annotated, and made model
        switches look like no-ops because both 8M and 650M would show
        the same Pfam-derived label. "—" makes the missing-annotation
        case visible; the Pfam info is already on the Domains layer.
        """
        if feat_idx in self._concepts:
            return self._concepts[feat_idx]
        return "—"

    def _load_done(self) -> None:
        if self._user_closing:
            return

        try:
            loading = self.query_one("#loading-label", Label)
        except Exception:
            return

        if self._error:
            short = self._error.splitlines()[0][:80] if self._error else ""
            loading.update(f"[red]Error:[/red] {short}")
            try:
                self.app.notify(
                    self._error, severity="error", timeout=30,
                )
            except Exception:
                pass
            return

        if self._concept_job:
            # Keep the label alive so concept progress stays visible.
            loading.update("[dim]Computing concepts…[/dim]")
            loading.display = True
        else:
            loading.update("")
            loading.display = False

        table = self.query_one("#feature-table", DataTable)
        table.display = True

        for rank, feat in enumerate(self._top_features, 1):
            table.add_row(
                str(rank),
                f"f/{feat['idx']}",
                self._concept_for_feature(feat["idx"]),
                f"{feat['max_act']:.3f}",
                key=str(feat["idx"]),
            )

        if self._top_features:
            table.move_cursor(row=0)
            self._show_feature(0)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_idx = event.cursor_row
        if row_idx is None or self._feature_acts is None:
            return
        self._show_feature(row_idx)

    def _show_feature(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self._top_features):
            return
        feat = self._top_features[row_idx]
        feat_idx = feat["idx"]
        acts_for_feature = self._feature_acts[:, feat_idx]  # (L,)

        try:
            self.query_one("#seq-pane", _SeqPane).set_feature(
                self._sequence, acts_for_feature
            )
        except Exception:
            pass

        if self._right_mode == "proteins" and self._sw_top_ids is not None:
            self._populate_proteins_table(feat_idx)

        try:
            self.query_one("#feature-label", Static).update(
                f"[bold]f/{feat_idx}[/bold]  [dim]layer {self._layer}[/dim]"
            )
        except Exception:
            pass

        try:
            threshold = 0.1
            n_above = int((acts_for_feature > threshold).sum())
            peak_pos = int(acts_for_feature.argmax()) + 1
            self.query_one("#feature-stats", Static).update(
                f"[dim]max {feat['max_act']:.3f} @ pos {peak_pos}  ·  "
                f"{n_above} residues > {threshold}[/dim]"
            )
        except Exception:
            pass

    def _populate_proteins_table(self, feat_idx: int) -> None:
        if self._sw_top_ids is None or self._sw_top_acts is None:
            return
        if feat_idx >= self._sw_top_ids.shape[0]:
            return
        try:
            table = self.query_one("#proteins-table", DataTable)
            table.clear()
            ids = self._sw_top_ids[feat_idx]
            acts = self._sw_top_acts[feat_idx].astype(float)
            for rank, (acc, v) in enumerate(zip(ids, acts), 1):
                if not acc:
                    break
                table.add_row(str(rank), acc, f"{v:.3f}")
        except Exception:
            pass

    def action_next_feature(self) -> None:
        try:
            table = self.query_one("#feature-table", DataTable)
            table.action_scroll_down()
        except Exception:
            pass

    def action_prev_feature(self) -> None:
        try:
            table = self.query_one("#feature-table", DataTable)
            table.action_scroll_up()
        except Exception:
            pass

    def action_compute_concepts(self) -> None:
        from .concepts_kb import ConceptsKBModal
        self.app.push_screen(
            ConceptsKBModal(self._model, self._layer, running_job=self._concept_job),
            self._on_concepts_kb_dismissed,
        )

    def _on_concepts_kb_dismissed(self, result) -> None:
        """Modal returns a dict {action, model, layer, n_proteins} or None."""
        if not result:
            return
        action = result.get("action")
        if action == "compute":
            # Refuse if any concept job is currently running anywhere —
            # `@work(exclusive=True)` would only cancel Textual's tracking,
            # not the remote process, so a second submission would race
            # with the first and overwrite its CSV.
            from ... import interplm
            running = [
                (m, l) for (m, l), rec in interplm.get_active_concepts_jobs().items()
                if rec.get("status") == "running"
            ]
            if running:
                m, l = running[0]
                cfg = interplm._MODELS.get(m, {})
                self.app.notify(
                    f"A concept job is already running "
                    f"({cfg.get('label', m)} layer {l}). "
                    "Wait for it to finish before starting another.",
                    severity="warning", timeout=8,
                )
                return
            self._set_loading_label("Connecting to remote…")
            self._compute_concepts_worker(
                result["model"], result["layer"], int(result["n_proteins"]),
            )
        elif action == "loaded":
            # Navigate to the selected (model, layer) so features and concepts
            # are always in sync — concepts are 1:1 with their model+layer.
            m = result.get("model", self._model)
            l = result.get("layer", self._layer)
            self._model = m
            self._layer = l
            self._reload()

    def on__model_layer_panel_selected(self, event: _ModelLayerPanel.Selected) -> None:
        if (event.model, event.layer) == (self._model, self._layer):
            return
        self._model = event.model
        self._layer = event.layer
        self._reload()

    def _navigate_and_refresh_concepts(
        self, model: str, layer: int, n_annotated: int
    ) -> None:
        """Switch to (model, layer) if needed, then populate the concept column."""
        from ... import interplm
        if (self._model, self._layer) != (model, layer):
            self._model = model
            self._layer = layer
            # _reload rebuilds the table from scratch with correct concepts.
            self._reload()
        else:
            self._concepts = interplm.load_concepts(layer, model)
            self._refresh_concepts_in_table(n_annotated)

    def _on_concept_started(self, model: str, layer: int) -> None:
        self._concept_job = (model, layer)
        self._update_subtitle()
        try:
            self.query_one("#model-layer-panel", _ModelLayerPanel).refresh()
        except Exception:
            pass

    def _on_concept_finished(self) -> None:
        self._concept_job = None
        self._update_subtitle()
        try:
            self.query_one("#model-layer-panel", _ModelLayerPanel).refresh()
        except Exception:
            pass
        try:
            lbl = self.query_one("#loading-label", Label)
            lbl.update("")
            lbl.display = False
        except Exception:
            pass

    @work(thread=True, exclusive=True, group="interplm-concepts")
    def _compute_concepts_worker(self, model: str, layer: int, n_proteins: int) -> None:
        from ... import interplm

        try:
            loop = self.app._loop
            if loop:
                loop.call_soon_threadsafe(self._on_concept_started, model, layer)
        except Exception:
            pass

        _phase = [""]  # mutable for closure

        def _push(msg: str) -> None:
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(self._set_loading_label, msg)
            except Exception:
                pass

        def _cb(raw: str) -> None:
            import re as _re
            line = raw.strip()
            if not line:
                return
            # "  N/M" pass progress lines
            m = _re.match(r"^\s*(\d+)/(\d+)\s*$", line)
            if m:
                i, total = int(m.group(1)), int(m.group(2))
                pct = int(100 * i / total) if total else 0
                filled = pct // 5
                bar = "█" * filled + "░" * (20 - filled)
                _push(f"{_phase[0]}: {i}/{total}  [{bar}] {pct}%")
                return
            # "  N/M proteins, …" fetch progress
            m2 = _re.match(r"^\s*(\d+)/(\d+)\s+proteins", line)
            if m2:
                i, total = int(m2.group(1)), int(m2.group(2))
                _push(f"Fetching proteins: {i}/{total}")
                return
            if "Fetching" in line and "proteins" in line:
                _phase[0] = "Fetching proteins"
                _push(line)
            elif "Fetched" in line and "proteins" in line:
                _push(line)
            elif "Loading ESM-2" in line or "Device:" in line:
                _push(line)
            elif "SAE loaded:" in line:
                _push(line)
            elif "Concepts kept:" in line:
                _push(line)
            elif "Pass 1/2" in line:
                _phase[0] = "Pass 1/2"
                _push("Pass 1/2: computing max activations…")
            elif "Pass 2/2" in line:
                _phase[0] = "Pass 2/2"
                _push("Pass 2/2: accumulating counters…")
            elif "Computing F1" in line:
                _phase[0] = "F1"
                _push("Computing F1 scores…")
            elif "Wrote " in line:
                _push(line)
            elif "done." in line:
                _push(line)
            elif not any(p in line for p in (
                "GPU available", "Existing torch", "Installing",
                "torch.cuda", "nvidia-smi",
            )):
                _push(line)

        try:
            n = interplm.compute_concepts_remote(
                layer=layer,
                model=model,
                n_proteins=n_proteins,
                progress_cb=_cb,
            )
            # Navigate to the computed (model, layer) so features and concepts
            # are 1:1 — then refresh the concept column.
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(
                        self._navigate_and_refresh_concepts, model, layer, n
                    )
            except Exception:
                pass
        except Exception as e:
            msg = f"Concept computation failed: {e}"
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(
                        lambda m=msg: self.app.notify(m, severity="error", timeout=30)
                    )
            except Exception:
                pass
        finally:
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(self._on_concept_finished)
            except Exception:
                pass

    def _refresh_concepts_in_table(self, n_annotated: int) -> None:
        if self._user_closing:
            return
        # Hide the loading label FIRST. If the table-rebuild raises, we
        # still want the streamed pip / progress chatter cleared from
        # the panel — otherwise the user is stuck looking at the last
        # remote stdout line forever.
        try:
            lbl = self.query_one("#loading-label", Label)
            lbl.update("")
            lbl.display = False
        except Exception:
            pass
        try:
            table = self.query_one("#feature-table", DataTable)
            # `add_columns(*labels)` generates auto column-keys, so
            # `update_cell(row_key, "Concept", …)` doesn't match. Rebuild
            # the table from scratch — cheap (≤200 rows) and avoids the
            # column-key plumbing entirely.
            table.clear()
            for rank, feat in enumerate(self._top_features, 1):
                table.add_row(
                    str(rank),
                    f"f/{feat['idx']}",
                    self._concept_for_feature(feat["idx"]),
                    f"{feat['max_act']:.3f}",
                    key=str(feat["idx"]),
                )
            if self._top_features:
                table.move_cursor(row=0)
        except Exception as e:
            self.app.notify(
                f"Table refresh error: {e}", severity="error", timeout=15,
            )
            return
        self.app.notify(
            f"Concepts loaded — {n_annotated} features annotated.", timeout=6,
        )

    def _update_subtitle(self) -> None:
        if self._concept_job:
            from ... import interplm
            cj_model, cj_layer = self._concept_job
            cj_cfg = interplm._MODELS.get(cj_model, {})
            label = cj_cfg.get("label", cj_model)
            self.sub_title = f"{self._project.name}  ·  computing concepts {label} L{cj_layer}"
        else:
            self.sub_title = self._project.name

    def _reload(self) -> None:
        from ... import interplm
        self._update_subtitle()
        try:
            self.query_one("#model-layer-panel", _ModelLayerPanel).set_selection(
                self._model, self._layer
            )
        except Exception:
            pass
        try:
            self._concepts = interplm.load_concepts(self._layer, self._model)
        except Exception:
            self._concepts = {}
        self._feature_acts = None
        self._top_features = []
        # Reset proteins pane for new layer/model
        self._sw_top_ids = None
        self._sw_top_acts = None
        self._right_mode = "sequence"
        try:
            self.query_one("#seq-pane", _SeqPane).display = True
            self.query_one("#proteins-table", DataTable).display = False
        except Exception:
            pass
        try:
            if interplm.has_swissprot_top100(self._layer, self._model):
                self._sw_top_ids, self._sw_top_acts = interplm.load_swissprot_top100(
                    self._layer, self._model
                )
        except Exception:
            pass
        try:
            table = self.query_one("#feature-table", DataTable)
            table.clear()
            table.display = False
        except Exception:
            pass
        self._set_loading_label("Loading…")
        self._load_async()

    def action_toggle_proteins(self) -> None:
        from ... import interplm
        if self._sw_top_ids is None:
            if interplm.has_swissprot_top100(self._layer, self._model):
                try:
                    self._sw_top_ids, self._sw_top_acts = interplm.load_swissprot_top100(
                        self._layer, self._model
                    )
                except Exception as e:
                    self.app.notify(f"Failed to load Swiss-Prot data: {e}", severity="error", timeout=8)
                    return
            else:
                self.app.notify(
                    "No Swiss-Prot data — press [p] to precompute (~20 min on GPU).",
                    timeout=5,
                )
                return

        self._right_mode = "proteins" if self._right_mode == "sequence" else "sequence"
        try:
            self.query_one("#seq-pane", _SeqPane).display = (self._right_mode == "sequence")
            self.query_one("#proteins-table", DataTable).display = (self._right_mode == "proteins")
        except Exception:
            pass

        if self._right_mode == "proteins":
            try:
                row = self.query_one("#feature-table", DataTable).cursor_row
                if row is not None and self._feature_acts is not None:
                    feat_idx = self._top_features[row]["idx"]
                    self._populate_proteins_table(feat_idx)
            except Exception:
                pass

    def action_precompute_swissprot(self) -> None:
        from ... import interplm
        if interplm.has_swissprot_top100(self._layer, self._model):
            self.app.notify(
                "Swiss-Prot top-100 already computed. Press [s] to view.",
                timeout=5,
            )
            return
        self._set_loading_label("Connecting to remote for Swiss-Prot precompute…")
        self._precompute_swissprot_worker()

    @work(thread=True, exclusive=True, group="interplm-swissprot")
    def _precompute_swissprot_worker(self) -> None:
        from ... import interplm

        def _cb(msg: str) -> None:
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(self._set_loading_label, msg)
            except Exception:
                pass

        try:
            interplm.compute_swissprot_remote(
                layer=self._layer,
                model=self._model,
                progress_cb=_cb,
            )
            top_ids, top_acts = interplm.load_swissprot_top100(self._layer, self._model)
            self._sw_top_ids = top_ids
            self._sw_top_acts = top_acts
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(self._on_swissprot_done)
            except Exception:
                pass
        except Exception as e:
            msg = f"Swiss-Prot precompute failed: {e}"
            try:
                loop = self.app._loop
                if loop:
                    loop.call_soon_threadsafe(
                        lambda m=msg: self.app.notify(m, severity="error", timeout=30)
                    )
            except Exception:
                pass

    def _on_swissprot_done(self) -> None:
        if self._user_closing:
            return
        try:
            lbl = self.query_one("#loading-label", Label)
            lbl.update("")
            lbl.display = False
        except Exception:
            pass
        self.app.notify("Swiss-Prot top-100 loaded — press [s] to view.", timeout=6)

    def action_back(self) -> None:
        self._user_closing = True
        self.app.pop_screen()
