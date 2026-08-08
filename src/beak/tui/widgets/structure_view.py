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
        """Posted when the target's CIF is on disk and parsed.

        `effective_mode` carries the color mode that was actually
        applied after `_on_loaded` validated `_color_mode` against the
        scalars that loaded successfully. Persisted prefs can request
        a mode whose data isn't available for the loaded structure
        (e.g. partial-coverage PDB with no conservation projection),
        in which case we silently fall back to b_iso. Surfacing the
        resolved mode lets the parent screen re-sync the dropdown
        without re-querying the StructureView's internals.

        `meta` carries the CIF header metadata (resolution, method)
        extracted by `read_cif_meta`. Empty dict for AlphaFold models
        or when the header couldn't be read.
        """
        def __init__(
            self,
            cif_path: Path,
            effective_mode: str = "plddt",
            meta: dict | None = None,
        ) -> None:
            super().__init__()
            self.cif_path = cif_path
            self.effective_mode = effective_mode
            self.meta: dict = meta if meta is not None else {}

    def __init__(self, project: BeakProject, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._coords = None
        self._plddt = None
        self._conservation = None
        self._sasa = None
        self._differential = None  # signed JSD per target residue
        self._taxonomic = None  # unsigned clade-bias per target residue [0,1]
        # Target-indexed copies (for chimerax export); see `_load`.
        self._conservation_target = None
        self._sasa_target = None
        self._differential_target = None
        self._taxonomic_target = None
        self._cif_seq: str = ""
        self._target_seq: str = ""
        # Per-residue Pfam index (-1 = outside any domain). Populated
        # lazily on demand via the manifest because the StructureView
        # otherwise has no reason to read the domains layer.
        self._pfam_idx = None
        self._cif_path: Optional[Path] = None
        self._struct_meta: dict = {}
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
        # Source label set by the loader (`AlphaFold` or `PDB <id>_<chain>`).
        # Drives the pLDDT-vs-B-factor adaptation in `_resolved_color_mode`
        # so the same dropdown entry renders the physically appropriate
        # quantity for whichever model is loaded.
        self._source_label: str = "AlphaFold"

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
        import os as _os
        self._refresh_subtitle()
        self._load()
        # Stash the timer so the parent screen can pause/resume it on
        # suspend. Set BEAK_NO_ROTATE=1 to skip starting the timer
        # entirely — useful as a triage tool if rotation-driven render
        # ticks are suspected of contributing to a freeze.
        if _os.environ.get("BEAK_NO_ROTATE") == "1":
            self._rotation_timer = None
            self._rotating = False
        else:
            self._rotation_timer = self.set_interval(
                1.0 / self._ROTATION_FPS, self._tick,
            )

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
        if mode == "taxonomic":
            self._reload_taxonomic()
            if self._taxonomic is None:
                return False
        if mode == "pfam":
            self._reload_pfam()
            if self._pfam_idx is None:
                return False
        self._color_mode = mode
        self._refresh_subtitle()
        self._refresh_canvas()
        return True

    def _effective_mode(self) -> str:
        """Resolve the user-facing mode to the physical quantity actually being rendered.

        The dropdown's `pLDDT` entry shows confidence (0–100) when an
        AlphaFold model is loaded but B-factor (Å², thermal motion)
        when a PDB experimental structure is loaded. Both live in the
        same cif column (`b_iso`) but mean opposite things, so we
        swap the rendering mode under the hood — palette, range,
        legend label, and ChimeraX export title all flip with the
        loaded source.
        """
        if self._color_mode == "plddt" and self._source_label.startswith("PDB"):
            return "bfactor"
        return self._color_mode

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
        """Pull the active comparative scores from disk; cheap on every flip.

        Stores both target-indexed (for export) and cif-indexed (for
        the canvas), the same convention as conservation / SASA in
        `_load`. Differential scores naturally land in target
        coordinates, so cif-projection happens on top of the raw
        load.
        """
        from ..structure import project_target_to_cif

        try:
            from ..comparative import load_active_scores
            raw = load_active_scores(self._project)
        except Exception:
            self._differential = None
            self._differential_target = None
            return
        self._differential_target = raw
        if raw is None or self._coords is None or not self._target_seq or not self._cif_seq:
            self._differential = raw  # AF case: cif and target align
            return
        projected = project_target_to_cif(
            raw, self._target_seq, self._cif_seq, sentinel=0.0,
        )
        self._differential = (
            projected if len(projected) == len(self._coords) else None
        )

    def has_differential(self) -> bool:
        if self._differential is None and self._differential_target is None:
            self._reload_differential()
        return self._differential is not None or self._differential_target is not None

    def _reload_taxonomic(self) -> None:
        """Pull active taxonomic-clustering scores; normalise to [0, 1].

        Mirrors `_reload_differential`, with one wrinkle: the score is
        either an uncertainty coefficient (already [0, 1]) or a
        permutation z-score (unbounded, possibly negative). For the ribbon
        we clip negatives and scale the z-scores to their own max so the
        gradient reads as relative clade-structure strength; the raw
        values still live in the cache for anyone who wants absolutes.
        """
        from ..structure import project_target_to_cif
        import numpy as np

        try:
            from ..comparative import load_active_taxonomic_scores
            raw = load_active_taxonomic_scores(self._project)
        except Exception:
            self._taxonomic = None
            self._taxonomic_target = None
            return
        if raw is None:
            self._taxonomic = None
            self._taxonomic_target = None
            return

        kind = (self._project.manifest().get("taxonomic") or {}).get(
            "active_score_kind"
        )
        disp = np.asarray(raw, dtype=float)
        if kind == "permutation_zscore":
            pos = np.clip(disp, 0.0, None)
            m = float(pos.max()) if pos.size else 0.0
            disp = pos / m if m > 0 else pos
        else:
            disp = np.clip(disp, 0.0, 1.0)

        self._taxonomic_target = disp
        if self._coords is None or not self._target_seq or not self._cif_seq:
            self._taxonomic = disp  # AF case: cif and target align
            return
        projected = project_target_to_cif(
            disp, self._target_seq, self._cif_seq, sentinel=0.0,
        )
        self._taxonomic = (
            projected if len(projected) == len(self._coords) else None
        )

    def has_taxonomic(self) -> bool:
        if self._taxonomic is None and self._taxonomic_target is None:
            self._reload_taxonomic()
        return self._taxonomic is not None or self._taxonomic_target is not None

    def reload_set_data(self) -> None:
        """Recompute set-scoped scalars (conservation/SASA/differential).

        Cheaper than a full reload — keeps the cached coords + pLDDT and
        only refreshes things that depend on which homolog set is
        active. Call this after the user switches sets.
        """
        if self._coords is None:
            return
        from ..structure import project_target_to_cif
        try:
            from ..conservation import compute_quick_conservation
            cons = compute_quick_conservation(self._project)
            self._conservation_target = cons
            if cons is not None and self._target_seq and self._cif_seq:
                projected = project_target_to_cif(
                    cons, self._target_seq, self._cif_seq, sentinel=0.0,
                )
                self._conservation = (
                    projected if len(projected) == len(self._coords) else None
                )
            else:
                self._conservation = None
        except Exception:
            self._conservation = None
            self._conservation_target = None
        self._reload_differential()
        self._reload_taxonomic()
        self._refresh_canvas()

    def export_current_to_chimerax(self, out_dir: Path, name: str) -> Optional[Tuple[Path, Path, int]]:
        """Write a CIF + .cxc for whatever scalar is currently rendered.

        Returns ``(cif_out, cxc_out, n_residues)`` on success, None when
        the underlying CIF or scalar isn't loaded yet.
        """
        cif_path = self._cif_path
        if cif_path is None or not cif_path.exists():
            return None
        target_seq = self._project.target_sequence() or ""
        if not target_seq:
            return None
        # Export wants target-indexed scalars (length = len(target_seq))
        # because `export_chimerax` does its own target → cif mapping
        # internally. Use `_scalar_for_export` (target-aligned) instead
        # of `_scalar_for_mode` (cif-aligned, used by the TUI canvas).
        scalars = self._scalar_for_export(target_seq)
        if scalars is None or len(scalars) != len(target_seq):
            return None
        from ...viz.chimerax import export_chimerax
        # Effective mode resolves "plddt → bfactor" when a PDB is
        # loaded — see `_effective_mode()`. The export title and
        # palette are picked off this resolved mode so the legend
        # advertises the right physical quantity (e.g. "B-factor (Å²)"
        # vs "AlphaFold pLDDT").
        eff_mode = self._effective_mode()
        title = f"{self._project.name} · {eff_mode}"
        if eff_mode == "differential":
            from ..comparative import active_label
            label = active_label(self._project)
            if label:
                title = f"{self._project.name} · {label}"
        elif eff_mode == "pfam":
            title = f"{self._project.name} · Pfam"
        # Pfam needs the full domain hit list from the manifest so the
        # export can emit per-domain color commands + a discrete legend
        # keyed on pfam_name. For every other mode the parameter stays
        # None and export_chimerax falls through to the generic
        # byattribute-palette path.
        pfam_domains = None
        if eff_mode == "pfam":
            pfam_domains = (
                self._project.manifest().get("domains") or {}
            ).get("hits") or None
        return export_chimerax(
            cif_path=cif_path,
            scores=scalars,
            target_seq=target_seq,
            out_dir=out_dir,
            name=name,
            mode=eff_mode,
            title=title,
            pfam_domains=pfam_domains,
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
        """Return the cif-indexed scalar feeding the TUI braille canvas."""
        if self._color_mode == "conservation" and self._conservation is not None:
            return self._conservation
        if self._color_mode == "sasa" and self._sasa is not None:
            return self._sasa
        if self._color_mode == "differential":
            if self._differential is None:
                self._reload_differential()
            if self._differential is not None:
                return self._differential
        if self._color_mode == "taxonomic":
            if self._taxonomic is None:
                self._reload_taxonomic()
            if self._taxonomic is not None:
                return self._taxonomic
        if self._color_mode == "pfam":
            if self._pfam_idx is None:
                self._reload_pfam()
            if self._pfam_idx is not None:
                return self._pfam_idx
        return self._plddt

    def _scalar_for_export(self, target_seq: str):
        """Return the *target-indexed* scalar for the chimerax export.

        The export module re-projects the target-indexed values onto
        the cif's residues itself (via `map_target_to_structure`), so
        callers must hand it data that's anchored to the target
        sequence — not the cif's residue order. For pLDDT / B-factor
        we project the cif b_iso back to target coords on demand,
        since the natural storage for those is cif-indexed.
        """
        from ..structure import project_target_to_cif
        # `_scalar_for_export` is only called once per export click,
        # so projecting here on demand is fine.
        if self._color_mode == "conservation":
            return self._conservation_target
        if self._color_mode == "sasa":
            return self._sasa_target
        if self._color_mode == "differential":
            if self._differential_target is None:
                self._reload_differential()
            return self._differential_target
        if self._color_mode == "taxonomic":
            if self._taxonomic_target is None:
                self._reload_taxonomic()
            return self._taxonomic_target
        if self._color_mode == "pfam":
            # Pfam idx is built per cif residue; we need target-indexed
            # for export. Read straight from manifest hits, which carry
            # target-residue ranges.
            domains = (self._project.manifest().get("domains") or {}).get("hits") or []
            if not domains:
                return None
            import numpy as np
            n = len(target_seq)
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
            return idx
        # Default = pLDDT / B-factor. Build target-indexed by walking
        # the cif residue numbers (1-based, SIFTS-aligned) and slotting
        # b_iso into the matching target position. AF cifs are 1:1 so
        # this collapses to the identity copy.
        if self._plddt is None or not self._cif_seq or not target_seq:
            return None
        import numpy as np
        # Reverse projection: align cif_seq to target_seq, walk paired
        # residues, copy cif b_iso → target position. Sentinel 0.0
        # for target positions outside the cif's coverage.
        out = np.zeros(len(target_seq), dtype=np.float32)
        try:
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner()
            aligner.mode = "global"
            aligner.match_score = 2
            aligner.mismatch_score = -1
            aligner.open_gap_score = -10
            aligner.extend_gap_score = -1
            best = aligner.align(target_seq, self._cif_seq)[0]
            t_blocks, c_blocks = best.aligned
        except Exception:
            return None
        for (t_start, t_end), (c_start, c_end) in zip(t_blocks, c_blocks):
            for tp, cp in zip(range(t_start, t_end), range(c_start, c_end)):
                if 0 <= tp < len(target_seq) and 0 <= cp < len(self._plddt):
                    out[tp] = float(self._plddt[cp])
        return out

    def has_conservation(self) -> bool:
        return self._conservation is not None or self._conservation_target is not None

    def has_sasa(self) -> bool:
        return self._sasa is not None or self._sasa_target is not None

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

        # Effective mode = `pLDDT → bfactor` swap when a PDB is loaded.
        # Used for both the gradient sample colors and the label text.
        eff_mode = self._effective_mode()

        # Categorical Pfam mode: paint one chip per actual domain so
        # the legend doubles as a mini-key. Capped so the chip strip
        # doesn't push past the available subtitle width on narrow
        # panels — the structure panel is 50–80 cells wide.
        if eff_mode == "pfam":
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

        # B-factor uses Å² with a different range (0–50) than the
        # 0–100 unitless modes. Pick the right range for the gradient
        # sweep so the bar is a faithful key for what's on screen.
        bar_cells = 10
        if eff_mode == "bfactor":
            bar_lo, bar_hi, hi_label = 0.0, 50.0, "50 Å²"
        elif eff_mode == "taxonomic":
            # Taxonomic scalars are normalised to [0, 1] by the view.
            bar_lo, bar_hi, hi_label = 0.0, 1.0, "max"
        else:
            bar_lo, bar_hi, hi_label = 0.0, 100.0, "100"
        cells = []
        for i in range(bar_cells):
            score = bar_lo + (i / (bar_cells - 1)) * (bar_hi - bar_lo)
            color = color_for_mode(score, eff_mode, self._midpoint)
            cells.append(f"[{color}]█[/{color}]")
        bar = "".join(cells)

        label = {
            "plddt": "pLDDT",
            "bfactor": "B-factor",
            "conservation": "cons",
            "sasa": "SASA",
            "differential": "diff",
            "taxonomic": "tax",
        }.get(eff_mode, eff_mode)

        if eff_mode == "conservation":
            head = f"[dim]{label} mid={int(self._midpoint)}[/dim]"
        else:
            head = f"[dim]{label}[/dim]"

        # Surface a stale-set badge in differential mode when the
        # cached scores were computed against a different homolog set
        # than is currently active. Mirrors the embeddings stale pill.
        if eff_mode == "differential":
            try:
                from ..comparative import differential_is_stale
                if differential_is_stale(self._project):
                    head = f"[yellow]{label} · stale[/yellow]"
            except Exception:
                pass
        if eff_mode == "taxonomic":
            try:
                from ..comparative import taxonomic_is_stale
                if taxonomic_is_stale(self._project):
                    head = f"[yellow]{label} · stale[/yellow]"
            except Exception:
                pass

        # Single-space gutter between the numeric labels and the bar so
        # they don't read as continuous with the gradient.
        legend = f"{head} [dim]{int(bar_lo)}[/dim] {bar} [dim]{hi_label}[/dim]"
        # Mirror the sequence panel's `[/]` keybinding hint — keeps the
        # midpoint adjuster discoverable on whichever panel the user
        # happens to be looking at. Only conservation cares about
        # midpoint, so suppress the hint elsewhere.
        if eff_mode == "conservation":
            legend += r" [dim]\[/\] adjust[/dim]"
        return legend

    def _refresh_canvas(self) -> None:
        try:
            self.query_one("#sv-canvas", _StructureCanvas).refresh()
        except Exception:
            pass

    def _tick(self) -> None:
        if not self._rotating or self._coords is None:
            return
        # Skip the frame if even a small backlog of writes is pending —
        # generating more rotation frames during pty back-pressure just
        # buries any user-interaction refresh behind stale animation.
        # Threshold of 3 frames is small enough that the rotation
        # yields to interactive work almost immediately; healthy ptys
        # drain in <1 frame so this never trips in normal operation.
        # Defensive try/except in case Textual reorganises the writer
        # internals.
        try:
            wq = self.app._driver._writer_thread._queue
            if wq.qsize() > 3:
                return
        except Exception:  # noqa: BLE001
            pass
        self._angle_y = (self._angle_y + self._DEGREES_PER_FRAME) % 360.0
        self._refresh_canvas()

    def pause_animation(self) -> None:
        """Pause the rotation timer (for screen suspend)."""
        timer = getattr(self, "_rotation_timer", None)
        if timer is not None:
            try:
                timer.pause()
            except Exception:
                pass
        # Defensive: if the user suspended the screen mid-drag (modal
        # popped a sub-screen), the down already captured the mouse
        # and the up will land on the new screen — never reaching our
        # canvas. Release here so capture doesn't outlive the drag.
        if self._dragging:
            try:
                canvas = self.query_one("#sv-canvas")
                canvas.release_mouse()
            except Exception:
                pass
            self._dragging = False
            self._rotating = self._was_rotating_before_drag

    def resume_animation(self) -> None:
        """Resume the rotation timer (for screen resume)."""
        timer = getattr(self, "_rotation_timer", None)
        if timer is not None:
            try:
                timer.resume()
            except Exception:
                pass

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
        # Always release capture, even if `_dragging` got reset by some
        # other path. If a button-up arrives without a matching down
        # (e.g. terminal dropped the press, the user alt-tabbed mid-
        # drag, screen popped between down and up), bailing out before
        # `release_mouse()` strands the canvas in capture-mode and
        # every subsequent click anywhere on screen is silently
        # swallowed by this widget.
        try:
            canvas.release_mouse()
        except Exception:
            pass
        if self._dragging:
            self._dragging = False
            self._rotating = self._was_rotating_before_drag

    # ---- async load ----

    @work(thread=True, exclusive=True, group="structure-load")
    def _load(self) -> None:
        from ..structure import (
            _cached_structure, find_or_fetch_structure,
            load_ca_residues, project_target_to_cif,
        )

        try:
            target = self._project.manifest().get("target", {}) or {}
            uniprot = target.get("uniprot_id")
            if not uniprot:
                self._status = "[dim]No structure: project has no UniProt ID.[/dim]"
                self.app.call_from_thread(self._refresh_canvas)
                return

            structures_dir = self._project.path / "structures"

            # Cache-first lookup avoids the SIFTS round-trip on every
            # project open. If nothing is cached, the second call kicks
            # off the (network-bound) PDB-then-AlphaFold resolution.
            # Honor a per-project preferred structure (set by the
            # structures-gallery "Make default" action).
            preferred = (
                (self._project.manifest().get("view") or {})
                .get("preferred_structure")
            )
            cached = _cached_structure(
                uniprot, structures_dir, preferred=preferred,
            )
            if cached is not None:
                cif_path, source_label = cached
            else:
                self._status = (
                    f"[dim]Looking up structure for {uniprot} "
                    f"(PDB → AlphaFold)…[/dim]"
                )
                self.app.call_from_thread(self._refresh_canvas)
                cif_path, source_label = find_or_fetch_structure(
                    uniprot, structures_dir,
                )

            coords, plddt, _res_nums, cif_seq = load_ca_residues(cif_path)
        except FileNotFoundError as e:
            self._status = f"[dim]{e}[/dim]"
            self.app.call_from_thread(self._refresh_canvas)
            return
        except Exception as e:  # noqa: BLE001
            self._status = f"[red]Structure error: {e}[/red]"
            self.app.call_from_thread(self._refresh_canvas)
            return

        target_seq = self._project.target_sequence() or ""

        # Each per-residue scalar is computed in its *natural* coordinate
        # system and then re-indexed for the consumers that need a
        # different one:
        #   * Conservation / differential come from the MSA → naturally
        #     indexed by target position (length = len(target_seq)).
        #   * SASA / B-factor come from the cif → naturally cif-indexed.
        #
        # The chimerax export wants target-indexed (it does its own
        # cif mapping internally). The TUI braille canvas iterates
        # `coords` (cif-indexed), so it wants scalars in the same
        # order. We keep a `_target` copy (for export) and a default
        # cif-aligned copy (for render). For an AlphaFold cif those
        # two are identical because the cif residues are 1:1 with
        # the target sequence; for a PDB cif with partial coverage
        # (e.g. residues 1-98 of a 99-aa target) the projection
        # trims / re-orders.

        # ---- Conservation ----
        cons_target = None
        conservation = None
        try:
            from ..conservation import compute_quick_conservation
            c = compute_quick_conservation(self._project)
            if c is not None and target_seq:
                cons_target = c
                conservation = project_target_to_cif(
                    c, target_seq, cif_seq, sentinel=0.0,
                )
                if len(conservation) != len(plddt):
                    conservation = None
                    cons_target = None
        except Exception:
            pass

        # ---- SASA ----
        # `load_sasa` is keyed by cif residue number. Pass target_len
        # so the returned array spans the full target — positions
        # outside the cif's coverage simply hold the sentinel (0.0)
        # and the projection step below assigns them to no cif
        # residue (since there is none).
        sasa_target = None
        sasa = None
        try:
            from ..structure import load_sasa
            n_for_load = max(len(target_seq) if target_seq else 0, len(plddt))
            s = load_sasa(cif_path, n_for_load) if n_for_load else None
            if s is not None and target_seq:
                sasa_target = s[: len(target_seq)] if len(s) >= len(target_seq) else s
                sasa = project_target_to_cif(
                    sasa_target, target_seq, cif_seq, sentinel=0.0,
                )
                if len(sasa) != len(plddt):
                    sasa = None
                    sasa_target = None
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
        # Cif-indexed (matches `coords`) versions for the TUI canvas.
        # These are the arrays the braille renderer iterates per-residue.
        self._plddt = plddt
        self._conservation = conservation
        self._sasa = sasa
        # Target-indexed copies for the chimerax export, which expects
        # length = len(target_seq) and does its own cif mapping. AF-loaded
        # projects have these equal to the cif-indexed copies; PDB-loaded
        # projects diverge when the experimental structure covers only
        # part of the target.
        self._conservation_target = cons_target
        self._sasa_target = sasa_target
        self._cif_seq = cif_seq
        self._target_seq = target_seq

        # Extract resolution / method from the CIF header for the info row.
        from ..structure import read_cif_meta
        meta = read_cif_meta(cif_path)

        self.app.call_from_thread(
            self._on_loaded, cif_path, uniprot, source_label, meta,
        )

    def _on_loaded(
        self,
        cif_path: Path,
        uniprot: str,
        source_label: str = "AlphaFold",
        meta: dict | None = None,
    ) -> None:
        # Worker → UI hop: the load worker fires `app.call_from_thread`
        # to land here, which can race the user popping the project
        # detail screen. Setting `border_title` and posting a message
        # on a vanished widget raises; guard the whole hop.
        try:
            self._cif_path = cif_path
            self._source_label = source_label
            self._struct_meta = meta if meta is not None else {}
            # Border reads e.g. "P00046 · PDB 2acp_A" or
            # "P00046 · AlphaFold" so the user can tell at a glance
            # which model the conservation/SASA/etc. is being mapped onto.
            self.border_title = f"{uniprot} · {source_label}"
            # Persisted `view.color_mode` can name a layer whose scalar
            # didn't load (typical case: PDB with partial coverage that
            # nukes the conservation projection). Fall back to b_iso so
            # the rendered surface matches the colorbar legend; the
            # CifLoaded message carries the resolved mode so the parent
            # screen can re-sync the dropdown to whichever mode actually
            # took.
            if not self._mode_data_available(self._color_mode):
                self._color_mode = "plddt"
            self._refresh_subtitle()
            self._refresh_canvas()
            self.post_message(
                self.CifLoaded(
                    cif_path,
                    effective_mode=self._color_mode,
                    meta=self._struct_meta,
                ),
            )
        except Exception:
            pass

    def _mode_data_available(self, mode: str) -> bool:
        """Whether the per-residue scalar for `mode` is loaded.

        b_iso always renders (it's the cif's intrinsic field); every
        other mode needs a backing array that may or may not have been
        derived during `_load`.
        """
        if mode == "plddt":
            return True
        if mode == "conservation":
            return self._conservation is not None
        if mode == "sasa":
            return self._sasa is not None
        if mode == "differential":
            return (
                self._differential is not None
                or self._differential_target is not None
            )
        if mode == "taxonomic":
            return (
                self._taxonomic is not None
                or self._taxonomic_target is not None
            )
        if mode == "pfam":
            domains = (
                self._project.manifest().get("domains") or {}
            ).get("hits") or []
            return bool(domains)
        return False

    # ---- rendering, called by inner _StructureCanvas.render ----

    def _render_canvas(self, w: int, h: int):
        """Build the structure-view canvas as a Rich `Text` (or markup str
        for status / empty cases).

        Returns a `rich.text.Text` for the rendered structure so the
        Textual compositor can skip the markup tokenizer entirely on
        every paint — the tokenizer regex was the source of a UI-freeze
        bug (catastrophic backtrack on ~thousands of `[#hex]` tags).
        Status placeholders return short markup strings; those parse in
        microseconds.
        """
        if self._coords is None:
            return self._status
        from math import radians
        from rich.text import Text
        from ..structure import render_structure
        if w < 4 or h < 2:
            return Text("")
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

        # Pull the canvas's actual rendered background so the mist fade
        # blends toward whatever bg the user picked (default / black /
        # slate / light / white). `styles.background` returns a Color
        # object (None when unset = inherit theme); `.hex` is the form
        # the renderer's mist helper accepts. Falls back to None on
        # any read failure so the renderer uses its own dim-gray
        # default.
        bg_hex = None
        try:
            canvas = self.query_one("#sv-canvas")
            bg_color = canvas.styles.background
            if bg_color is not None:
                bg_hex = bg_color.hex
        except Exception:
            pass

        rendered = render_structure(
            self._coords, scalar, w, render_h,
            angle_y=radians(self._angle_y),
            angle_x=radians(self._angle_x),
            color_mode=self._effective_mode(),
            midpoint=self._midpoint,
            view_mode=self._view_mode,
            bg_color=bg_hex,
        )

        if show_arrow:
            arrow = self._midpoint_arrow_row(w)
            # `rendered` is a Rich Text now; concatenate via Text.append.
            rendered.append("\n")
            rendered.append(arrow)
        return rendered

    def _midpoint_arrow_row(self, width: int):
        """Single canvas row with `▼` aligned over the bar's mid cell.

        Layout assumed for the right-aligned `border_subtitle`:
            ...{label} 0 [10-cell bar] 100 [/\\] adjust
        Working back from the subtitle's right edge:
          - "adjust"           — 6 chars
          - " "                — 1
          - "[/\\]"            — 4 (Rich only escapes `\\[`, not `\\]`)
          - " "                — 1
          - "100"              — 3
          - " "                — 1
        That's 16 chars after the bar. The subtitle's rightmost char
        sits at the same column as the canvas's last column (border +
        no padding either side), so the bar's rightmost cell lands at
        canvas col `width - 1 - 16 = width - 17` and its leftmost at
        `width - 26`.
        """
        from rich.style import Style
        from rich.text import Text
        bar_cells = 10
        # ` [/\] adjust` (12 chars) only renders in conservation mode.
        # Other modes that show the arrow would offset differently,
        # but conservation is the only mode this row is drawn for.
        suffix_after_bar = 16  # 1 + 3 + 1 + 4 + 1 + 6
        bar_left = width - 1 - suffix_after_bar - (bar_cells - 1)
        # Cell index closest to the current midpoint (0..bar_cells-1).
        mid_idx = round((self._midpoint / 100.0) * (bar_cells - 1))
        mid_idx = max(0, min(bar_cells - 1, mid_idx))
        arrow_col = bar_left + mid_idx
        if not 0 <= arrow_col < width:
            return Text(" " * width)
        text = Text()
        text.append(" " * arrow_col)
        text.append("▼", style=Style(color="#65CBF3"))
        text.append(" " * (width - arrow_col - 1))
        return text
