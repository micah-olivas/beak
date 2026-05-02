"""Project detail screen — target metadata, layer state, structure, sequence."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Select, Static

from ...project import BeakProject
from ..widgets.colorbar import Colorbar
from ..widgets.layers_panel import LayersPanel
from ..widgets.sequence_view import SequenceView
from ..widgets.structure_view import StructureView


class ProjectDetailScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "view_alignment", "Alignment"),
        Binding("p", "view_pca", "PCA"),
        Binding("t", "view_taxonomy", "Taxonomy"),
        Binding("d", "toggle_domains", "Domains"),
        Binding("s", "toggle_ss", "SS"),
        # Search → Layers panel pill. Color → sequence-panel dropdown.
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="detail-body"):
            with Vertical(id="info-col"):
                yield Static(self._target_panel(), classes="panel", id="target-panel")
                yield LayersPanel(self._project, classes="panel", id="layers-panel")
            with Vertical(id="struct-col"):
                yield StructureView(
                    self._project, classes="panel", id="structure-view"
                )
                with Horizontal(id="struct-controls"):
                    yield Label("Color:")
                    yield Select(
                        [("pLDDT", "plddt"),
                         ("Conservation", "conservation"),
                         ("SASA", "sasa"),
                         ("Differential", "differential")],
                        value=self._color_mode,
                        id="color-select",
                        allow_blank=False,
                        compact=True,
                    )
                    yield Label("View:")
                    yield Select(
                        [("trace", "trace"), ("tube", "tube")],
                        value=self._view_mode,
                        id="view-select",
                        allow_blank=False,
                        compact=True,
                    )
                    yield Label("BG:")
                    yield Select(
                        [("default", "default"), ("black", "black"),
                         ("slate", "slate"), ("light", "light"),
                         ("white", "white")],
                        value=self._bg_choice,
                        id="bg-select",
                        allow_blank=False,
                        compact=True,
                    )
                    # Export button — writes CIF + .cxc for the active mode.
                    yield Button("Export", id="export-cxc-btn", compact=True)
                # `struct-colorbar` is yielded *inside* StructureView so
                # the gradient sits adjacent to the ribbon it's coloring.
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

    def action_refresh(self) -> None:
        self.query_one("#target-panel", Static).update(self._target_panel())
        # Layers panel: refresh local + kick off a remote poll if any jobs are pending.
        self.query_one(LayersPanel).force_refresh()
        self.query_one(SequenceView).reload()
        self.notify("Refreshed")

    def on_structure_view_cif_loaded(self, message: StructureView.CifLoaded) -> None:
        # Structure CIF just landed on disk — pull SS + pLDDT into the sequence view.
        self.query_one(SequenceView).reload()

    def action_submit_search(self) -> None:
        from .submit_search import SubmitSearchModal
        self.app.push_screen(SubmitSearchModal(self._project), self._on_search_submitted)

    def on_colorbar_midpoint_changed(self, message) -> None:
        # Forward midpoint change to the structure view + the *other*
        # colorbar so all three surfaces stay in sync.
        self.query_one(StructureView).set_midpoint(message.midpoint)
        for cb in self.query(Colorbar):
            if cb.midpoint != message.midpoint:
                cb.set_midpoint(message.midpoint)
        self.query_one(SequenceView).set_midpoint(message.midpoint)

    def on_select_changed(self, event) -> None:
        sid = getattr(event.select, "id", None)
        sv = self.query_one(StructureView)
        if sid == "color-select":
            mode = str(event.value)
            if mode == self._color_mode:
                return
            seq = self.query_one(SequenceView)
            if not sv.set_color_mode(mode):
                with event.select.prevent(Select.Changed):
                    event.select.value = self._color_mode
                return
            seq.set_color_mode(mode)
            try:
                self.query_one("#struct-colorbar", Colorbar).set_mode(mode)
            except Exception:
                pass
            self._color_mode = mode
            self._save_view_pref("color_mode", mode)
        elif sid == "view-select":
            mode = str(event.value)
            sv.set_view_mode(mode)
            self._view_mode = mode
            self._save_view_pref("view_mode", mode)
        elif sid == "bg-select":
            choice = str(event.value)
            self._apply_bg(sv, choice)
            self._bg_choice = choice
            self._save_view_pref("bg", choice)

    def on_mount(self) -> None:
        sv = self.query_one(StructureView)
        self._apply_bg(sv, self._bg_choice)
        if self._view_mode != "trace":
            sv.set_view_mode(self._view_mode)
        # Sync the structure-side colorbar to persisted mode.
        try:
            self.query_one("#struct-colorbar", Colorbar).set_mode(self._color_mode)
        except Exception:
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
            m = self._project.manifest()
            view = m.setdefault("view", {})
            view[key] = value
            self._project.write(m)
        except Exception:
            pass  # don't let a manifest write failure break the UI

    def on_layer_row_clicked(self, message) -> None:
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
        if action_str == "open-search":
            self.action_submit_search()
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
        elif message.pill_id == "import-pill":
            from .import_experiment import ImportExperimentModal
            self.app.push_screen(
                ImportExperimentModal(self._project), self._on_experiment_imported
            )
        elif message.pill_id in ("homologs-status-pill", "embeddings-status-pill") and message.value:
            from .job_status import JobStatusModal
            self.app.push_screen(JobStatusModal(message.value))

    def _on_tax_submitted(self, job_id) -> None:
        if not job_id:
            return
        self.notify(f"Taxonomy submitted: {job_id}", timeout=6)
        self.query_one(LayersPanel).refresh_state()

    def _on_embed_submitted(self, job_id) -> None:
        if not job_id:
            return
        self.notify(f"Embeddings submitted: {job_id}", timeout=8)
        self.query_one(LayersPanel).refresh_state()

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

    def _on_search_submitted(self, job_id) -> None:
        if not job_id:
            return
        self.notify(f"Search submitted: {job_id}", timeout=8)
        self.query_one(LayersPanel).refresh_state()
