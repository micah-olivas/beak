"""Project detail screen — target metadata, layer state, structure, sequence."""

from pathlib import Path
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Select, Static

from ...project import BeakProject
from ..widgets.layers_panel import LayersPanel
from ..widgets.sequence_view import SequenceView
from ..widgets.server_status import ServerStatusBar
from ..widgets.structure_view import StructureView


class ProjectDetailScreen(Screen):
    # Don't auto-focus a child widget on mount.
    #
    # The default behaviour (`AUTO_FOCUS = "*"` on App) picks the first
    # focusable descendant, which on this screen lands on the
    # color-select Select. Newer Textual versions treat focused Select
    # widgets as keyboard-search receivers — every printable letter
    # gets consumed for option lookup, so the screen's `r/a/p/t/d/s/q`
    # bindings never fire. The user can still navigate by clicking,
    # but key bindings appear "frozen". Mouse-clicking elsewhere
    # moves focus off the Select and bindings start working again.
    #
    # `AUTO_FOCUS = None` doesn't disable auto-focus — Textual treats
    # `None` as "fall back to the App's AUTO_FOCUS". The empty string
    # is the sentinel that actually skips it. Confirmed via the
    # SIGUSR1 widget-tree dump (focused=Select(color-select)).
    AUTO_FOCUS = ""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("e", "edit_project", "Edit"),
        Binding("a", "view_alignment", "Alignment"),
        Binding("p", "view_pca", "PCA"),
        Binding("t", "view_taxonomy", "Taxonomy"),
        Binding("i", "view_interplm", "InterPLM"),
        Binding("d", "toggle_domains", "Domains"),
        Binding("s", "toggle_ss", "SS"),
        # Conservation midpoint shift — replaces the inline colorbar's
        # ◀/▶ keys now that the legend lives on the panel border. Shown
        # in the Footer only when conservation is the active color mode
        # (see `check_action`); a `[/]` hint also rides along on the
        # gradient legend itself for at-a-glance discovery.
        Binding("[", "midpoint_down", "−mid"),
        Binding("]", "midpoint_up",   "+mid"),
    ]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        prefs = self._project.manifest().get("view") or {}
        # Views read the persisted mode in their own __init__ now; the
        # screen tracks it too so on_select_changed has somewhere to compare.
        self._color_mode: str = prefs.get("color_mode", "plddt")
        view = prefs.get("view_mode", "trace")
        self._view_mode: str = "trace" if view == "shaded" else view
        self._bg_choice: str = prefs.get("bg", "default")
        # Mirrors the value the views render against. Owned here (not on
        # a child widget) because the inline focusable Colorbar is gone
        # and the keybindings on the screen are the only adjuster.
        self._midpoint: float = 50.0
        self._resume_animation_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        # Thin 1-row strip below the header — content right-aligned via
        # the widget's own CSS, so the empty left side just blends with
        # the screen background.
        yield ServerStatusBar(id="server-status")
        with Horizontal(id="detail-body"):
            with Vertical(id="info-col"):
                yield Static(self._target_panel(), classes="panel", id="target-panel")
                yield LayersPanel(self._project, classes="panel", id="layers-panel")
            with Vertical(id="struct-col"):
                yield StructureView(
                    self._project, classes="panel", id="structure-view"
                )
                # All four controls share one row. Labels are minimal
                # ("color/view/bg") and the dropdown options are
                # abbreviated so the Export button stays right-aligned
                # without pushing past the structure column at narrow
                # aspect ratios.
                with Horizontal(id="struct-controls"):
                    yield Label("color", classes="ctrl-lbl")
                    yield Select(
                        [("pLDDT", "plddt"),
                         ("cons", "conservation"),
                         ("SASA", "sasa"),
                         ("diff", "differential"),
                         ("tax", "taxonomic"),
                         ("Pfam", "pfam")],
                        value=self._color_mode,
                        id="color-select",
                        allow_blank=False,
                        compact=True,
                    )
                    yield Label("view", classes="ctrl-lbl")
                    yield Select(
                        [("trace", "trace"), ("tube", "tube")],
                        value=self._view_mode,
                        id="view-select",
                        allow_blank=False,
                        compact=True,
                    )
                    yield Label("bg", classes="ctrl-lbl")
                    yield Select(
                        [("default", "default"), ("black", "black"),
                         ("slate", "slate"), ("light", "light"),
                         ("white", "white")],
                        value=self._bg_choice,
                        id="bg-select",
                        allow_blank=False,
                        compact=True,
                    )
                    # `Try PDB` upgrades an AlphaFold-defaulted project to
                    # the best experimental structure available via SIFTS.
                    # `variant="primary"` makes it visually distinct from
                    # the plain `Export` next to it (otherwise the two
                    # render as indistinguishable compact text). Hidden
                    # entirely once the loaded structure is already PDB —
                    # see `on_structure_view_cif_loaded`.
                    yield Button(
                        "Try PDB",
                        id="try-pdb-btn",
                        variant="primary",
                        compact=True,
                    )
                    # Export button — writes CIF + .cxc for the active mode.
                    yield Button("Export", id="export-cxc-btn", compact=True)
                yield Static("", id="struct-meta")
        yield SequenceView(self._project, classes="panel", id="sequence-view")
        yield Footer()

    def _target_panel(self) -> str:
        m = self._project.manifest()
        target = m.get("target", {})
        proj_meta = m.get("project", {})
        lines = [f"[bold]{self._project.name}[/bold]"]
        if proj_meta.get("description"):
            lines.append(f"[dim]{proj_meta['description']}[/dim]")
        lines.append("")
        for label, key in (("UniProt", "uniprot_id"), ("Gene", "gene_name"),
                           ("Organism", "organism"), ("Length", "length")):
            val = target.get(key)
            if val is not None:
                lines.append(f"  [bold]{label}:[/bold] {val}")
        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-cxc-btn":
            self._export_chimerax()
        elif event.button.id == "try-pdb-btn":
            self._try_pdb()

    def _try_pdb(self) -> None:
        """Look up an experimental PDB structure for the target via SIFTS.

        Worker-driven so the SSL handshake to PDBe doesn't freeze the
        TUI. The button stays clickable; spamming it just queues
        another worker (the work decorator's exclusive=True drops the
        duplicate immediately). Cached AF is preserved either way —
        if a PDB is already on disk the worker reports it; if SIFTS
        returns nothing the toast says so explicitly.
        """
        target = self._project.manifest().get("target", {}) or {}
        uniprot = target.get("uniprot_id")
        if not uniprot:
            self.notify(
                "Try PDB needs a UniProt ID on the target — none recorded.",
                severity="warning", timeout=6,
            )
            return
        self.notify("Looking up PDB structure…", timeout=4)
        self._try_pdb_worker(uniprot)

    @work(thread=True, exclusive=True, group="try-pdb")
    def _try_pdb_worker(self, uniprot: str) -> None:
        from ...api.structures import find_structures, fetch_structures
        from ..structure import _cached_structure

        struct_dir = self._project.path / "structures"
        cached = _cached_structure(uniprot, struct_dir)
        if cached is not None and cached[1].startswith("PDB"):
            # Already on PDB — clicking the button is a no-op, but be
            # explicit so the user isn't left guessing.
            self.app.call_from_thread(
                self.notify,
                f"Already using {cached[1]}. Delete that file to "
                f"fall back to AlphaFold.",
                timeout=8,
            )
            return

        try:
            df = find_structures([uniprot], source="pdb")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"PDB lookup failed: {type(e).__name__}: {e}",
                severity="error", timeout=8,
            )
            return

        if df is None or df.empty:
            self.app.call_from_thread(
                self.notify,
                f"No PDB structure found for {uniprot} — keeping AlphaFold.",
                timeout=8,
            )
            return

        try:
            fetched = fetch_structures(
                df, output_dir=str(struct_dir),
                selection="best", skip_existing=True,
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self.notify,
                f"PDB download failed: {type(e).__name__}: {e}",
                severity="error", timeout=8,
            )
            return

        ok = fetched[fetched["local_path"].notna()]
        if ok.empty:
            err = fetched["error"].dropna().head(1).iloc[0] if not fetched["error"].dropna().empty else "unknown"
            self.app.call_from_thread(
                self.notify,
                f"PDB download produced no files (error: {err}). "
                f"Keeping AlphaFold.",
                severity="warning", timeout=8,
            )
            return

        row = ok.iloc[0]
        struct_id = row["structure_id"]
        chain = row.get("chain_id") or ""
        label = f"PDB {struct_id}_{chain}" if chain and chain != "-" else f"PDB {struct_id}"

        # Reload the structure-view from the main thread — `reload()`
        # is what kicks `_load` again, which now picks up the freshly
        # cached PDB ahead of any AF cif sitting in the same dir.
        self.app.call_from_thread(self._reload_structure_view)
        self.app.call_from_thread(
            self.notify,
            f"Switched to {label}. (AlphaFold cif kept — delete it to "
            f"manage cache.)",
            timeout=8,
        )

    def _reload_structure_view(self) -> None:
        try:
            sv = self.query_one(StructureView)
        except Exception:
            return
        # `_load` is decorated with @work(exclusive=True), so calling
        # it again cancels any in-flight rotation/render and reloads
        # cleanly.
        sv._load()

    def _export_chimerax(self) -> None:
        """Hand the current scalar + CIF off to the ChimeraX exporter."""
        sv = self.query_one(StructureView)
        out_dir = self._project.path / "exports" / "chimerax"
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self._project.name}__{self._color_mode}__{stamp}"
        try:
            result = sv.export_current_to_chimerax(out_dir, name)
        except Exception as e:  # noqa: BLE001
            self.notify(f"Export failed: {e}", severity="error", timeout=8)
            return
        if result is None:
            self.notify(
                "Nothing to export — structure or scalar isn't loaded yet.",
                severity="warning", timeout=6,
            )
            return
        cif_out, cxc_out, n_set = result
        try:
            rel = cxc_out.relative_to(self._project.path)
        except ValueError:
            rel = cxc_out

        from ...viz.chimerax import open_in_chimerax
        opened: bool
        try:
            binary = open_in_chimerax(cxc_out)
            opened = True
        except Exception:
            opened = False
            binary = None
        suffix = (
            "  · opening in ChimeraX…" if opened
            else "  · ChimeraX not found ($CHIMERAX to set path)"
        )
        self.notify(
            f"Export ready · {rel}  ({n_set} residues){suffix}", timeout=10
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_edit_project(self) -> None:
        from .rename_project import RenameProjectModal
        self.app.push_screen(
            RenameProjectModal(self._project), self._on_project_info_saved
        )

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "target-panel":
            self.action_edit_project()
            event.stop()

    def _on_project_info_saved(self, new_name) -> None:
        if new_name is None:
            return
        self.query_one("#target-panel", Static).update(self._target_panel())

    def action_refresh(self) -> None:
        self.query_one("#target-panel", Static).update(self._target_panel())
        # Layers panel: refresh local + kick off a remote poll if any jobs are pending.
        self.query_one(LayersPanel).force_refresh()
        self.query_one(SequenceView).reload()
        self.notify("Refreshed")

    @staticmethod
    def _format_struct_meta(source_label: str, meta: dict) -> str:
        """Build the one-line dim info string for the #struct-meta row."""
        if source_label.startswith("AlphaFold"):
            return "[dim]AlphaFold · pLDDT confidence model[/dim]"
        parts = []
        res = meta.get("resolution")
        if res is not None:
            parts.append(f"{res:.1f} Å")
        method = meta.get("method")
        if method:
            parts.append(method)
        if parts:
            return "[dim]" + " · ".join(parts) + "[/dim]"
        return ""

    def on_structure_view_cif_loaded(self, message: StructureView.CifLoaded) -> None:
        # Structure CIF just landed on disk — pull SS + pLDDT into the sequence view.
        # If the StructureView fell back to b_iso (because the persisted
        # mode's scalar didn't load), align the screen + sequence view
        # to the same effective mode so all three render the same thing.
        if message.effective_mode != self._color_mode:
            self._color_mode = message.effective_mode
            try:
                self.query_one(SequenceView).set_color_mode(message.effective_mode)
            except Exception:
                pass
        self.query_one(SequenceView).reload()
        # Populate the structure metadata info row.
        try:
            sv = self.query_one(StructureView)
            info_text = self._format_struct_meta(sv._source_label, message.meta)
            self.query_one("#struct-meta", Static).update(info_text)
        except Exception:
            pass
        # Reflect the loaded structure's source in the controls row:
        try:
            sv = self.query_one(StructureView)
            is_pdb = sv._source_label.startswith("PDB")
        except Exception:
            return
        # 1. Hide the `Try PDB` button once we're already on a PDB
        #    structure — it would only re-fetch what's already cached.
        try:
            self.query_one("#try-pdb-btn", Button).display = not is_pdb
        except Exception:
            pass
        # 2. Flip the color-mode dropdown's label for the b_iso entry —
        #    `pLDDT` for AlphaFold confidence, `B-factor` for PDB
        #    crystallographic temperature factor. The internal mode
        #    value stays `"plddt"` so persisted view-prefs keep
        #    working; only the user-visible label changes.
        biso_label = "B-factor" if is_pdb else "pLDDT"
        self._sync_color_dropdown(biso_label=biso_label)

    def _sync_color_dropdown(
        self, *, biso_label: Optional[str] = None,
    ) -> None:
        """Re-paint the color dropdown so its label + selected entry track
        the structure view's actual mode.

        Two responsibilities, kept together so callers don't have to
        reason about Textual's `Select` reactive quirks:

        * Re-build the option list with the correct b_iso label
          (pLDDT for AlphaFold, B-factor for PDB). When `biso_label`
          is None we infer it from the current StructureView source,
          so callers that just want to re-sync after a color-mode
          change don't have to know which structure is loaded.
        * Set the visible value to whatever `self._color_mode` says,
          updating SelectCurrent directly. Textual's `_watch_value`
          only repaints when the reactive `value` actually changes,
          which fails us in two ways:
            - After `set_options`, value resets to the first option
              ("plddt"). If the screen's `_color_mode` was already
              "plddt", the watcher never fires and the SelectCurrent
              keeps painting the pre-`set_options` text.
            - If a non-dropdown code path (toggle, comparative-built)
              advanced `_color_mode`, the dropdown was never updated
              to match.
          Reaching into SelectCurrent.update sidesteps both.
        """
        from textual.widgets import Select
        from textual.widgets._select import SelectCurrent
        try:
            sel = self.query_one("#color-select", Select)
        except Exception:
            return
        if biso_label is None:
            try:
                sv = self.query_one(StructureView)
                biso_label = (
                    "B-factor" if sv._source_label.startswith("PDB") else "pLDDT"
                )
            except Exception:
                biso_label = "pLDDT"
        options = [
            (biso_label, "plddt"),
            ("cons", "conservation"),
            ("SASA", "sasa"),
            ("diff", "differential"),
            ("tax", "taxonomic"),
            ("Pfam", "pfam"),
        ]
        sel.set_options(options)
        # Set the underlying value (suppress Changed so on_select_changed
        # doesn't try to apply a mode the user didn't pick).
        with sel.prevent(Select.Changed):
            sel.value = self._color_mode
        # Force the visible prompt to repaint regardless of whether
        # the watcher fired. Look up the matching option's prompt and
        # push it directly into SelectCurrent.
        try:
            prompt = next(p for p, v in options if v == self._color_mode)
        except StopIteration:
            return
        try:
            sel.query_one(SelectCurrent).update(prompt)
        except Exception:
            pass

    def action_submit_search(self) -> None:
        from .submit_search import SubmitSearchModal
        self.app.push_screen(SubmitSearchModal(self._project), self._on_search_submitted)

    def check_action(self, action: str, parameters):
        # Hide the midpoint keys from the Footer (and refuse the keypress)
        # whenever they wouldn't do anything — every other color mode
        # ignores midpoint, so showing `[/]` for them would be noise.
        if action in ("midpoint_down", "midpoint_up"):
            return self._color_mode == "conservation"
        return True

    def action_midpoint_down(self) -> None:
        self._shift_midpoint(-5.0)

    def action_midpoint_up(self) -> None:
        self._shift_midpoint(+5.0)

    def _shift_midpoint(self, delta: float) -> None:
        # Conservation is the only mode whose color depends on midpoint;
        # ignore the keypress in any other mode rather than visibly
        # changing a value that doesn't affect the display.
        if self._color_mode != "conservation":
            return
        self._midpoint = max(5.0, min(95.0, self._midpoint + delta))
        self.query_one(StructureView).set_midpoint(self._midpoint)
        self.query_one(SequenceView).set_midpoint(self._midpoint)

    def on_select_changed(self, event) -> None:
        sid = getattr(event.select, "id", None)
        sv = self.query_one(StructureView)
        if sid == "color-select":
            mode = str(event.value)
            if mode == self._color_mode:
                return
            seq = self.query_one(SequenceView)
            if not sv.set_color_mode(mode):
                # Mode unavailable (e.g. conservation before homologs
                # land). Snap the dropdown back to whatever's actually
                # being rendered — `_sync_color_dropdown` updates
                # SelectCurrent directly so the visible prompt can't
                # drift past Textual's reactive watcher.
                self._sync_color_dropdown()
                return
            seq.set_color_mode(mode)
            self._color_mode = mode
            self._save_view_pref("color_mode", mode)
            # Re-evaluate `check_action` so the Footer drops/restores
            # the `[/]` hint as the user toggles into/out of conservation.
            self.refresh_bindings()
        elif sid == "view-select":
            mode = str(event.value)
            # Each Select posts an initial Changed when its value is set
            # in `compose`, so this handler runs three times on every
            # detail-screen mount. Early-return when the value matches
            # what we already had — without this, mount triggers
            # spurious `set_view_mode` + `_save_view_pref` work that
            # cascades into Footer rebuilds and disk writes.
            if mode == self._view_mode:
                return
            sv.set_view_mode(mode)
            self._view_mode = mode
            self._save_view_pref("view_mode", mode)
        elif sid == "bg-select":
            choice = str(event.value)
            if choice == self._bg_choice:
                return
            self._apply_bg(sv, choice)
            self._bg_choice = choice
            self._save_view_pref("bg", choice)
            # Mist fades toward the canvas bg, so the structure needs
            # a re-render whenever bg changes — otherwise the haze
            # stays tinted from the previous theme.
            sv._refresh_canvas()

    def on_mount(self) -> None:
        sv = self.query_one(StructureView)
        self._apply_bg(sv, self._bg_choice)
        if self._view_mode != "trace":
            sv.set_view_mode(self._view_mode)

    def on_screen_suspend(self) -> None:
        # Fully pause the structure-view rotation timer while a child
        # screen (alignment, PCA, taxonomy) covers us. We previously
        # had a per-tick `is_active` guard, but the bare timer firing
        # at 10 fps coincided with input-thread wedges on macOS after
        # popping back. Pausing the timer outright eliminates that
        # whole class of timing interaction.
        timer = getattr(self, "_resume_animation_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._resume_animation_timer = None
        try:
            self.query_one(StructureView).pause_animation()
        except Exception:
            pass

    def on_unmount(self) -> None:
        timer = getattr(self, "_resume_animation_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._resume_animation_timer = None

    def on_screen_resume(self) -> None:
        # Defer structure rotation until the screen-pop repaint has
        # settled. Restarting the 10 fps canvas timer in the same tick
        # as the alignment-view pop can add stale rotation frames to the
        # writer queue before the user-visible detail repaint drains.
        try:
            captured = self.app.mouse_captured
            if captured is not None:
                captured.release_mouse()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._resume_animation_timer = self.set_timer(
                0.35, self._resume_structure_animation_when_idle,
            )
        except Exception:  # noqa: BLE001
            self._resume_structure_animation_when_idle()

    def _resume_structure_animation_when_idle(self) -> None:
        self._resume_animation_timer = None
        # If the terminal writer is still draining a transition burst,
        # wait another beat. This keeps animation frames from competing
        # with the first stable detail-screen repaint.
        try:
            wq = self.app._driver._writer_thread._queue
            if wq.qsize() > 3:
                self._resume_animation_timer = self.set_timer(
                    0.35, self._resume_structure_animation_when_idle,
                )
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            self.query_one(StructureView).resume_animation()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _apply_bg(sv, choice: str) -> None:
        # Apply bg to the inner canvas only — the outer StructureView
        # widget's bounds extend slightly past what the panel border
        # encloses, so setting bg there caused visible bleed below the
        # bottom border. The canvas has explicit `height: 1fr` inside
        # the panel and stays within the visible content box.
        colors = {
            "black": "#000000",
            "slate": "#1A1F2E",
            "light": "#4A4F58",
            "white": "#F5F5F7",
        }
        try:
            canvas = sv.query_one("#sv-canvas")
        except Exception:
            return
        if choice in colors:
            canvas.styles.background = colors[choice]
        else:
            canvas.styles.background = None

    def _save_view_pref(self, key: str, value) -> None:
        try:
            with self._project.mutate() as m:
                view = m.setdefault("view", {})
                view[key] = value
        except Exception:
            pass  # don't let a manifest write failure break the UI

    def on_layer_row_clicked(self, message) -> None:
        if message.layer_name == "features":
            from .interplm_view import InterPLMScreen
            self.app.push_screen(
                InterPLMScreen(self._project),
                lambda _: self.query_one(LayersPanel).refresh_state(),
            )
            return
        from .layer_detail import LayerDetailModal
        self.app.push_screen(
            LayerDetailModal(self._project, message.layer_name),
            self._on_layer_modal_dismissed,
        )

    def _on_layer_modal_dismissed(self, action) -> None:
        if not action:
            return
        action_str = str(action)
        if action_str == "set-switched":
            # Active set changed — refresh everything that reads from it.
            self.query_one(LayersPanel).refresh_state()
            self.query_one(SequenceView).reload()
            # Conservation, SASA, and differential scalars are scoped to
            # the active set; refresh the structure view so the ribbon
            # coloring tracks the new set without refetching the CIF.
            self.query_one(StructureView).reload_set_data()
            self.notify(
                f"Switched to set: {self._project.active_set_name()}",
                timeout=4,
            )
            return
        if action_str == "tax-rebuilt":
            self.query_one(LayersPanel).refresh_state()
            return
        if action_str == "set-deleted":
            self.query_one(LayersPanel).refresh_state()
            self.query_one(SequenceView).reload()
            self.query_one(StructureView).reload_set_data()
            return
        if action_str == "set-renamed":
            # The renamed set may also be the active one — refresh
            # anything that resolves the active set's directory.
            self.query_one(LayersPanel).refresh_state()
            self.query_one(SequenceView).reload()
            self.query_one(StructureView).reload_set_data()
            return
        if action_str == "set-mutated":
            # In-modal mutations (filter, etc.) flushed dirty state on
            # close — refresh the same surfaces a switch/rename would.
            self.query_one(LayersPanel).refresh_state()
            self.query_one(SequenceView).reload()
            self.query_one(StructureView).reload_set_data()
            return
        if action_str == "structure-default-changed":
            # User picked a new preferred structure in the gallery —
            # `reload_set_data` keeps the old CIF cached, so call the
            # full `_load` worker to swap in the chosen file. Sequence
            # view re-reads pLDDT/SS from the new CIF on `CifLoaded`.
            self._reload_structure_view()
            self.query_one(LayersPanel).refresh_state()
            return
        if action_str.startswith("submit-align:"):
            # Modal asked us to (re-)submit alignment for a specific set.
            # `LayersPanel.submit_alignment` already owns the worker +
            # status pill; we just hand off the set name.
            target_set = action_str.split(":", 1)[1]
            self.query_one(LayersPanel).submit_alignment(target_set)
            return
        if action_str == "open-search":
            self.action_submit_search()
            return
        if action_str == "open-embed":
            from .submit_embed import SubmitEmbedModal
            self.app.push_screen(
                SubmitEmbedModal(self._project), self._on_embed_submitted
            )
            return
        if action_str.startswith("reset"):
            self.query_one(LayersPanel).refresh_state()
            self.query_one(SequenceView).reload()
            self.notify(f"Reset {action_str.replace('reset-', '')}", timeout=4)

    def on_pill_pressed(self, message) -> None:
        # Single dispatch for the layers-panel pills.
        if message.pill_id == "search-pill":
            self.action_submit_search()
        elif message.pill_id == "align-pill":
            self.query_one(LayersPanel).submit_alignment()
        elif message.pill_id == "tax-pill":
            from .submit_taxonomy import SubmitTaxonomyModal
            self.app.push_screen(
                SubmitTaxonomyModal(self._project), self._on_tax_submitted
            )
        elif message.pill_id == "embed-pill":
            from .submit_embed import SubmitEmbedModal
            self.app.push_screen(
                SubmitEmbedModal(self._project), self._on_embed_submitted
            )
        elif message.pill_id == "diff-pill":
            from .submit_comparative import SubmitComparativeModal
            self.app.push_screen(
                SubmitComparativeModal(self._project),
                self._on_comparative_built,
            )
        elif message.pill_id == "tax-cluster-pill":
            from .submit_taxonomic import SubmitTaxonomicModal
            self.app.push_screen(
                SubmitTaxonomicModal(self._project),
                self._on_taxonomic_built,
            )
        elif message.pill_id == "features-pill":
            from .interplm_view import InterPLMScreen
            self.app.push_screen(
                InterPLMScreen(self._project),
                lambda _: self.query_one(LayersPanel).refresh_state(),
            )
        elif message.pill_id == "import-pill":
            from .import_experiment import ImportExperimentModal
            self.app.push_screen(
                ImportExperimentModal(self._project), self._on_experiment_imported
            )
        elif message.pill_id in ("homologs-status-pill", "embeddings-status-pill") and message.value:
            from .job_status import JobStatusModal
            # Refresh the layers panel after the modal closes so a
            # successful cancel/clear from inside the modal flips the
            # row state without waiting for the next poll. The modal
            # gets the project so it can clear failed jobs out of the
            # manifest (which is what re-enables the Embed/Search/Align
            # pills after a failure).
            self.app.push_screen(
                JobStatusModal(message.value, project=self._project),
                lambda _: self.query_one(LayersPanel).force_refresh(),
            )

    def _on_tax_submitted(self, job_id) -> None:
        if not job_id:
            return
        self.notify(f"Taxonomy submitted: {job_id}", timeout=6)
        self.query_one(LayersPanel).refresh_state()

    def _on_embed_submitted(self, params) -> None:
        # Modal now returns a params dict (or None on cancel). The
        # remote `mgr.submit()` call runs here in a background worker
        # so the user can keep working while the (potentially slow,
        # multi-minute) first-time container build runs on the remote.
        if not params:
            return
        self.notify(
            f"Submitting embedding job · {params.get('model')}…",
            timeout=4,
        )
        # Capture the App on the UI thread before dispatching to the
        # worker. @work(thread=True) doesn't reliably propagate the
        # `active_app` contextvar into the worker thread when the
        # worker is launched from a modal-dismiss callback — without
        # this we hit NoActiveAppError on the first `self.app` access
        # inside the worker (after a successful submit, no less, so
        # the toast for the job_id was getting eaten). Passing the
        # app reference explicitly sidesteps the contextvar entirely.
        self._run_embed_submit(self.app, params)

    @work(thread=True, exclusive=True, group="embed-submit")
    def _run_embed_submit(self, app, params: dict) -> None:
        from ...remote.embeddings import ESMEmbeddings
        mgr = None
        try:
            mgr = ESMEmbeddings()
            job_id = mgr.submit(
                params["hits_fasta"],
                model=params["model"],
                job_name=params["job_name"],
                repr_layers=params["layers"],
            )
            self._project.update_active_embeddings_set(
                model=params["model"],
                remote={"job_id": job_id},
            )
            app.call_from_thread(
                self.notify,
                f"Embeddings submitted: {job_id}",
                timeout=8,
            )
            # `query_one` is a UI-thread operation; calling it from
            # the worker (even as the argument to `call_from_thread`)
            # raises NoMatches when the screen has been popped between
            # submit and now. Schedule a tiny UI-thread function that
            # itself does the lookup + refresh, with its own guard.
            app.call_from_thread(self._refresh_layers_panel_safely)
        except Exception as e:  # noqa: BLE001
            self._handle_embed_submit_failure(app, params, e)
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass

    def _refresh_layers_panel_safely(self) -> None:
        """UI-thread refresh of the layers panel that survives a
        screen pop. Used by the embed-submit worker after success
        — we don't want a navigation race to fail the whole submit
        with a NoMatches traceback."""
        try:
            self.query_one(LayersPanel).refresh_state()
        except Exception:
            pass

    def _handle_embed_submit_failure(self, app, params: dict, exc: BaseException) -> None:
        """Write the full error to a log file and surface a toast that
        points the user to it. Without this, the only signal the user
        had was a one-line truncated error in a frozen modal — the
        Docker `--- stderr ---` section that actually pinpoints the
        cause was clipped off the right edge.
        """
        import traceback
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path.home() / ".beak" / "errors"
        log_path = log_dir / f"{ts}_embed_submit_{self._project.name}.log"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"# beak embed submit failure · {ts}\n")
                f.write(f"# project: {self._project.name}\n")
                f.write(f"# model:   {params.get('model')}\n")
                f.write(f"# layers:  {params.get('layers')}\n")
                f.write(f"# fasta:   {params.get('hits_fasta')}\n")
                f.write(f"\n{type(exc).__name__}: {exc}\n\n")
                f.write("--- traceback ---\n")
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__, file=f,
                )
        except Exception:
            log_path = None

        msg = (
            f"Embed submission failed: {type(exc).__name__}: "
            f"{str(exc)[:120]}"
        )
        if log_path is not None:
            msg += f"\nFull log: {log_path}"
        app.call_from_thread(
            self.notify, msg, severity="error", timeout=15,
        )

    def _on_comparative_built(self, column) -> None:
        if not column:
            return
        sv = self.query_one(StructureView)
        if sv.set_color_mode("differential"):
            self._color_mode = "differential"
            self._save_view_pref("color_mode", "differential")
            try:
                self.query_one("#color-select", Select).value = "differential"
            except Exception:
                pass
            try:
                seq = self.query_one(SequenceView)
                seq.set_color_mode("differential")
            except Exception:
                pass
            from ..comparative import active_label
            label = active_label(self._project) or column
            self.notify(f"Differential: {label}", timeout=6)
        else:
            self.notify(
                "Differential built but couldn't load — try toggling Color "
                "to 'Differential'.",
                severity="warning", timeout=8,
            )
        self.query_one(LayersPanel).refresh_state()

    def _on_taxonomic_built(self, rank) -> None:
        if not rank:
            return
        sv = self.query_one(StructureView)
        if sv.set_color_mode("taxonomic"):
            self._color_mode = "taxonomic"
            self._save_view_pref("color_mode", "taxonomic")
            try:
                self.query_one("#color-select", Select).value = "taxonomic"
            except Exception:
                pass
            try:
                seq = self.query_one(SequenceView)
                seq.set_color_mode("taxonomic")
            except Exception:
                pass
            from ..comparative import active_taxonomic_label
            label = active_taxonomic_label(self._project) or rank
            self.notify(f"Taxonomic clustering: {label}", timeout=6)
        else:
            self.notify(
                "Taxonomic scores built but couldn't load — try toggling "
                "Color to 'tax'.",
                severity="warning", timeout=8,
            )
        self.query_one(LayersPanel).refresh_state()

    def _on_experiment_imported(self, name) -> None:
        if not name:
            return
        self.notify(f"Imported experiment: {name}", timeout=6)
        self.query_one(LayersPanel).refresh_state()

    def action_view_alignment(self) -> None:
        from .alignment_view import AlignmentViewerScreen
        self.app.push_screen(AlignmentViewerScreen(self._project))

    def action_view_pca(self) -> None:
        from .embedding_pca import EmbeddingPCAScreen
        self.app.push_screen(EmbeddingPCAScreen(self._project))

    def action_view_taxonomy(self) -> None:
        from .taxonomy_view import TaxonomyViewerScreen, has_taxonomy
        if not has_taxonomy(self._project):
            self.notify(
                "Taxonomy not yet annotated — run a search and wait for "
                "the taxonomy join to complete.",
                severity="warning", timeout=6,
            )
            return
        self.app.push_screen(TaxonomyViewerScreen(self._project))

    def action_view_interplm(self) -> None:
        from .interplm_view import InterPLMScreen
        self.app.push_screen(InterPLMScreen(self._project))

    def action_toggle_domains(self) -> None:
        # Keybinding shortcut — the panel pill is the discoverable affordance.
        self.query_one(SequenceView).toggle_domains()

    def action_toggle_ss(self) -> None:
        self.query_one(SequenceView).toggle_ss()

    def action_toggle_color(self) -> None:
        next_mode = "conservation" if self._color_mode == "plddt" else "plddt"
        sv = self.query_one(StructureView)
        seq = self.query_one(SequenceView)
        if not (sv.set_color_mode(next_mode) and seq.set_color_mode(next_mode)):
            return  # silent — conservation isn't available yet
        self._color_mode = next_mode
        # Keep the dropdown in lockstep — without this the visible
        # prompt drifts away from the actual rendered mode the moment
        # any non-click path advances `_color_mode`.
        self._sync_color_dropdown()

    def _on_search_submitted(self, job_id) -> None:
        if not job_id:
            return
        self.notify(f"Search submitted: {job_id}", timeout=8)
        self.query_one(LayersPanel).refresh_state()
