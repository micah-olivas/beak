"""StructureView — braille ribbon panel with header toggles.

Loads the AlphaFold model for the project's UniProt ID (fetching on
first view) and renders Cα coordinates as braille characters colored
by a per-residue scalar (pLDDT or conservation).

Layout: a header row with a clickable color-mode pill, plus the
braille canvas below it. The canvas owns mouse events for drag-rotate.

Worker → state → render() pattern: the load worker only writes state and
calls refresh() on the canvas — calling self.update() from a worker thread
with heavy markup content trips a Textual 8.x bug.
"""

from pathlib import Path
from typing import Optional, Tuple

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from ...project import BeakProject
from .colorbar import Colorbar


class _StructureCanvas(Static):
    """Inner braille-rendering body. Holds no state; queries the parent."""

    ALLOW_SELECT = False

    def __init__(self, parent_view: "StructureView", **kwargs) -> None:
        super().__init__("[dim]Loading structure…[/dim]", **kwargs)
        self._pv = parent_view

    def render(self):
        return self._pv._render_canvas(self.size.width, self.size.height)

    def on_resize(self, event) -> None:
        if self._pv._coords is not None:
            self.refresh()

    def on_mouse_down(self, event) -> None:
        self._pv._canvas_mouse_down(self, event)

    def on_mouse_move(self, event) -> None:
        self._pv._canvas_mouse_move(event)

    def on_mouse_up(self, event) -> None:
        self._pv._canvas_mouse_up(self, event)


class StructureView(Vertical):
    """Outer panel: header pill row + braille canvas + state."""

    DEFAULT_CSS = """
    StructureView { height: auto; }
    /* The canvas fills the panel; the gradient is overlaid in the
       bottom-right by `_render_canvas`, no separate widget needed. */
    StructureView _StructureCanvas {
        height: 1fr;
    }
    """

    # Rotation: ~12 s per revolution (3°/frame at 10 fps).
    _ROTATION_FPS = 10
    _DEGREES_PER_FRAME = 3.0

    # Drag sensitivity: degrees per cell.
    _DRAG_DEG_PER_CELL_X = 5.0
    _DRAG_DEG_PER_CELL_Y = 8.0
    _MAX_TILT = 90.0

    class CifLoaded(Message):
        """Posted when the target's CIF is on disk and parsed."""
        def __init__(self, cif_path: Path) -> None:
            super().__init__()
            self.cif_path = cif_path

    def __init__(self, project: BeakProject, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._coords = None
        self._plddt = None
        self._conservation = None
        self._sasa = None
        self._differential = None  # signed JSD per target residue
        # Per-residue Pfam index (-1 = outside any domain). Populated
        # lazily on demand via the manifest because the StructureView
        # otherwise has no reason to read the domains layer.
        self._pfam_idx = None
        self._cif_path: Optional[Path] = None
        # Honor persisted view prefs from the project manifest so the first
        # render after re-opening a project matches the user's last choice.
        prefs = (project.manifest().get("view") or {})
        self._color_mode: str = prefs.get("color_mode", "plddt")
        self._midpoint: float = 50.0
        legacy_view = prefs.get("view_mode", "trace")
        self._view_mode: str = "trace" if legacy_view == "shaded" else legacy_view
        self._status: str = "[dim]Loading structure…[/dim]"
        self._angle_y: float = 0.0
        self._angle_x: float = 0.0
        self._rotating: bool = True
        self._dragging: bool = False
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._drag_start_angle_y: float = 0.0
        self._drag_start_angle_x: float = 0.0
        self._was_rotating_before_drag: bool = True

        # Size the panel: width tracks protein length; height stays
        # tight so the sequence panel below gets enough room for
        # multi-block layouts on long proteins.
        target = (project.manifest().get("target") or {})
        length = int(target.get("length") or 200)
        self.styles.width = max(50, min(80, 30 + length // 20))
        self.styles.height = max(18, min(22, 16 + length // 60))

    def compose(self) -> ComposeResult:
        yield _StructureCanvas(self, id="sv-canvas")
        # No separate colorbar widget — the gradient is overlaid by
        # `_render_canvas` in the bottom-right corner of the canvas.

    def on_mount(self) -> None:
        self._refresh_subtitle()
        self._load()
        self.set_interval(1.0 / self._ROTATION_FPS, self._tick)

    def set_color_mode(self, mode: str) -> bool:
        """Switch the per-residue color scalar. Returns True if applied."""
        if mode == "conservation" and self._conservation is None:
            return False
        if mode == "sasa" and self._sasa is None:
            return False
        if mode == "differential":
            self._reload_differential()
            if self._differential is None:
                return False
        if mode == "pfam":
            self._reload_pfam()
            if self._pfam_idx is None:
                return False
        self._color_mode = mode
        self._refresh_subtitle()
        self._refresh_canvas()
        return True

    def _reload_pfam(self) -> None:
        """Build the per-residue Pfam index from the manifest's hmmscan hits."""
        if self._coords is None:
            self._pfam_idx = None
            return
        domains = (self._project.manifest().get("domains") or {}).get("hits") or []
        if not domains:
            self._pfam_idx = None
            return
        import numpy as np
        n = len(self._coords)
        idx = np.full(n, -1, dtype=np.int8)
        for i, d in enumerate(domains):
            try:
                start = max(0, int(d.get("env_from", 1)) - 1)
                end = min(n, int(d.get("env_to", 0)))
            except (TypeError, ValueError):
                continue
            for pos in range(start, end):
                if idx[pos] == -1:
                    idx[pos] = i
        self._pfam_idx = idx

    def has_pfam(self) -> bool:
        if self._pfam_idx is None:
            self._reload_pfam()
        return self._pfam_idx is not None

    def _reload_differential(self) -> None:
        """Pull the active comparative scores from disk; cheap on every flip."""
        try:
            from ..comparative import load_active_scores
            self._differential = load_active_scores(self._project)
        except Exception:
            self._differential = None

    def has_differential(self) -> bool:
        if self._differential is None:
            self._reload_differential()
        return self._differential is not None

    def reload_set_data(self) -> None:
        """Recompute set-scoped scalars (conservation/SASA/differential).

        Cheaper than a full reload — keeps the cached coords + pLDDT and
        only refreshes things that depend on which homolog set is
        active. Call this after the user switches sets.
        """
        if self._coords is None:
            return
        try:
            from ..conservation import compute_quick_conservation
            cons = compute_quick_conservation(self._project)
            self._conservation = (
                cons if cons is not None and len(cons) == len(self._coords) else None
            )
        except Exception:
            self._conservation = None
        self._reload_differential()
        self._refresh_canvas()

    def export_current_to_chimerax(self, out_dir: Path, name: str) -> Optional[Tuple[Path, Path, int]]:
        """Write a CIF + .cxc for whatever scalar is currently rendered.

        Returns ``(cif_out, cxc_out, n_residues)`` on success, None when
        the underlying CIF or scalar isn't loaded yet.
        """
        cif_path = self._cif_path
        if cif_path is None or not cif_path.exists():
            return None
        scalars = self._scalar_for_mode()
        if scalars is None:
            return None
        target_seq = self._project.target_sequence() or ""
        if not target_seq or len(scalars) != len(target_seq):
            return None
        from ...viz.chimerax import export_chimerax
        title = f"{self._project.name} · {self._color_mode}"
        if self._color_mode == "differential":
            from ..comparative import active_label
            label = active_label(self._project)
            if label:
                title = f"{self._project.name} · {label}"
        return export_chimerax(
            cif_path=cif_path,
            scores=scalars,
            target_seq=target_seq,
            out_dir=out_dir,
            name=name,
            mode=self._color_mode,
            title=title,
        )

    def set_midpoint(self, midpoint: float) -> None:
        self._midpoint = midpoint
        # The legend lives in the border subtitle and shows `mid=NN` for
        # conservation mode, so it has to refresh alongside the canvas.
        self._refresh_subtitle()
        self._refresh_canvas()

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        self._refresh_canvas()

    def _scalar_for_mode(self):
        if self._color_mode == "conservation" and self._conservation is not None:
            return self._conservation
        if self._color_mode == "sasa" and self._sasa is not None:
            return self._sasa
        if self._color_mode == "differential":
            if self._differential is None:
                self._reload_differential()
            if self._differential is not None:
                return self._differential
        if self._color_mode == "pfam":
            if self._pfam_idx is None:
                self._reload_pfam()
            if self._pfam_idx is not None:
                return self._pfam_idx
        return self._plddt

    def has_conservation(self) -> bool:
        return self._conservation is not None

    def has_sasa(self) -> bool:
        return self._sasa is not None

    def _refresh_subtitle(self) -> None:
        """Render the colorbar legend on the panel's bottom border.

        Living on the border (rather than overlaying the canvas's last
        row) tucks the legend literally into the corner of the panel
        outline and frees up a row of structure pixels. Min/max labels
        come along for free, mirroring the inline sequence-view legend.
        """
        self.border_subtitle = self._build_legend()

    def _build_legend(self) -> str:
        from ..structure import color_for_mode, DOMAIN_PALETTE

        # Categorical Pfam mode: paint one chip per actual domain so
        # the legend doubles as a mini-key. Capped so the chip strip
        # doesn't push past the available subtitle width on narrow
        # panels — the structure panel is 50–80 cells wide.
        if self._color_mode == "pfam":
            domains = (
                self._project.manifest().get("domains") or {}
            ).get("hits") or []
            if not domains:
                return ""
            cells = []
            for i in range(min(len(domains), 8)):
                color = DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
                cells.append(f"[{color}]█[/{color}]")
            return "[dim]Pfam[/dim] " + "".join(cells)

        bar_cells = 10
        cells = []
        for i in range(bar_cells):
            score = (i / (bar_cells - 1)) * 100
            color = color_for_mode(score, self._color_mode, self._midpoint)
            cells.append(f"[{color}]█[/{color}]")
        bar = "".join(cells)

        label = {
            "plddt": "pLDDT",
            "conservation": "cons",
            "sasa": "SASA",
            "differential": "diff",
        }.get(self._color_mode, self._color_mode)

        if self._color_mode == "conservation":
            head = f"[dim]{label} mid={int(self._midpoint)}[/dim]"
        else:
            head = f"[dim]{label}[/dim]"

        # Surface a stale-set badge in differential mode when the
        # cached scores were computed against a different homolog set
        # than is currently active. Mirrors the embeddings stale pill.
        if self._color_mode == "differential":
            try:
                from ..comparative import differential_is_stale
                if differential_is_stale(self._project):
                    head = f"[yellow]{label} · stale[/yellow]"
            except Exception:
                pass

        # Single-space gutter between the numeric labels and the bar so
        # they don't read as continuous with the gradient.
        return f"{head} [dim]0[/dim] {bar} [dim]100[/dim]"

    def _refresh_canvas(self) -> None:
        try:
            self.query_one("#sv-canvas", _StructureCanvas).refresh()
        except Exception:
            pass

    def _tick(self) -> None:
        if not self._rotating or self._coords is None:
            return
        self._angle_y = (self._angle_y + self._DEGREES_PER_FRAME) % 360.0
        self._refresh_canvas()

    # ---- mouse events forwarded by the canvas ----

    def _canvas_mouse_down(self, canvas, event) -> None:
        if self._coords is None:
            return
        canvas.capture_mouse()
        self._dragging = True
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_angle_y = self._angle_y
        self._drag_start_angle_x = self._angle_x
        self._was_rotating_before_drag = self._rotating
        self._rotating = False
        event.stop()

    def _canvas_mouse_move(self, event) -> None:
        if not self._dragging:
            return
        delta_x = event.x - self._drag_start_x
        delta_y = event.y - self._drag_start_y
        self._angle_y = (
            self._drag_start_angle_y + delta_x * self._DRAG_DEG_PER_CELL_X
        ) % 360.0
        new_tilt = self._drag_start_angle_x + delta_y * self._DRAG_DEG_PER_CELL_Y
        self._angle_x = max(-self._MAX_TILT, min(self._MAX_TILT, new_tilt))
        self._refresh_canvas()

    def _canvas_mouse_up(self, canvas, event) -> None:
        if not self._dragging:
            return
        canvas.release_mouse()
        self._dragging = False
        self._rotating = self._was_rotating_before_drag

    # ---- async load ----

    @work(thread=True, exclusive=True, group="structure-load")
    def _load(self) -> None:
        from ..structure import fetch_alphafold, load_ca_coords

        try:
            target = self._project.manifest().get("target", {}) or {}
            uniprot = target.get("uniprot_id")
            if not uniprot:
                self._status = "[dim]No structure: project has no UniProt ID.[/dim]"
                self.app.call_from_thread(self._refresh_canvas)
                return

            structures_dir = self._project.path / "structures"
            existing = (
                sorted(structures_dir.glob(f"{uniprot}_AF.cif"))
                if structures_dir.exists() else []
            )

            if existing:
                cif_path = existing[0]
            else:
                self._status = f"[dim]Fetching AlphaFold model for {uniprot}…[/dim]"
                self.app.call_from_thread(self._refresh_canvas)
                cif_path = fetch_alphafold(uniprot, structures_dir)

            coords, plddt = load_ca_coords(cif_path)
        except FileNotFoundError as e:
            self._status = f"[dim]{e}[/dim]"
            self.app.call_from_thread(self._refresh_canvas)
            return
        except Exception as e:  # noqa: BLE001
            self._status = f"[red]Structure error: {e}[/red]"
            self.app.call_from_thread(self._refresh_canvas)
            return

        # Quick per-residue conservation from hits.fasta when present.
        conservation = None
        try:
            from ..conservation import compute_quick_conservation
            cons = compute_quick_conservation(self._project)
            if cons is not None and len(cons) == len(plddt):
                conservation = cons
        except Exception:
            pass

        # Per-residue SASA from the CIF.
        sasa = None
        try:
            from ..structure import load_sasa
            s = load_sasa(cif_path, len(plddt))
            if s is not None and len(s) == len(plddt):
                sasa = s
        except Exception:
            pass

        # PCA-align so the protein's longest axis is horizontal at the
        # default rotation — fills the panel width nicely from the start.
        try:
            import numpy as np
            centered = coords - coords.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            # Ensure right-handed coordinate system to avoid mirror-image
            if np.linalg.det(vt) < 0:
                vt[2] *= -1
            coords = centered @ vt.T
        except Exception:
            pass

        self._coords = coords
        self._plddt = plddt
        self._conservation = conservation
        self._sasa = sasa
        self.app.call_from_thread(self._on_loaded, cif_path, uniprot)

    def _on_loaded(self, cif_path: Path, uniprot: str) -> None:
        self._cif_path = cif_path
        self.border_title = f"{uniprot} · AlphaFold"
        self._refresh_subtitle()
        self._refresh_canvas()
        self.post_message(self.CifLoaded(cif_path))

    # ---- rendering, called by inner _StructureCanvas.render ----

    def _render_canvas(self, w: int, h: int) -> str:
        if self._coords is None:
            return self._status
        from math import radians
        from ..structure import render_structure
        if w < 4 or h < 2:
            return ""
        scalar = self._scalar_for_mode()

        # Conservation is the only mode with a user-adjustable midpoint,
        # so it's the only one that earns a `▼` indicator. We reserve the
        # last canvas row for the arrow and render the structure into
        # the row above so the `▼` sits directly over the bar that lives
        # in `border_subtitle` one row below.
        show_arrow = (
            self._color_mode == "conservation"
            and h >= 3 and w >= 14
        )
        render_h = h - 1 if show_arrow else h

        rendered = render_structure(
            self._coords, scalar, w, render_h,
            angle_y=radians(self._angle_y),
            angle_x=radians(self._angle_x),
            color_mode=self._color_mode,
            midpoint=self._midpoint,
            view_mode=self._view_mode,
        )

        if show_arrow:
            rendered += "\n" + self._midpoint_arrow_row(w)
        return rendered

    def _midpoint_arrow_row(self, width: int) -> str:
        """Single canvas row with `▼` aligned over the bar's mid cell.

        Layout assumed for `border_subtitle`:
            ... 0 [10-cell bar] 100
        With Textual right-aligning the subtitle (default) and reserving
        one cell for the bottom-right corner glyph, the bar's rightmost
        cell sits 4 cells in from the panel's right edge ("100" + space
        + corner). Translating to canvas coordinates (which are inset by
        the panel's 2-cell horizontal padding + 1-cell border), the bar
        occupies canvas cols `width-12` through `width-3`.
        """
        bar_cells = 10
        # Cell index closest to the current midpoint (0..bar_cells-1).
        mid_idx = round((self._midpoint / 100.0) * (bar_cells - 1))
        mid_idx = max(0, min(bar_cells - 1, mid_idx))
        arrow_col = (width - 12) + mid_idx
        if not 0 <= arrow_col < width:
            return " " * width
        return (
            " " * arrow_col
            + "[#65CBF3]▼[/#65CBF3]"
            + " " * (width - arrow_col - 1)
        )
