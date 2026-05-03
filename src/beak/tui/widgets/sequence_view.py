"""SequenceView — target sequence with secondary structure annotation.

Wraps the sequence to panel width. Above the sequence:
    - SS markers (H/E in red/yellow)
    - Pfam domain bars (auto-shown whenever hmmscan returned hits)

Sequence letters are colored by the same per-residue scalar as the
structure viewport (pLDDT for now), so colors line up across views.
The 'd' keybinding on the project screen toggles domain visibility.
"""

from typing import List, Optional

import numpy as np
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from ...project import BeakProject
from .colorbar import Colorbar


_SS_COLORS = {
    'H': '#FF6B6B',
    'E': '#FFD93D',
}

# Cycle through these for sequential Pfam domains. The structure view
# uses the same palette (via `tui.structure.DOMAIN_PALETTE`) so a
# residue's domain reads as the same color whether you're looking at
# the ribbon or the sequence panel.
from ..structure import DOMAIN_PALETTE as _DOMAIN_PALETTE


class _SequenceBody(Static):
    """Inner Static that renders the wrapped sequence text."""

    def __init__(self, parent_view: "SequenceView", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._pv = parent_view

    def render(self):
        return self._pv._render_body(self.size.width)


class SequenceView(Vertical):
    """Sequence panel: header pill row + body Static."""

    DEFAULT_CSS = """
    /* The panel takes whatever vertical room the parent allocates; the
       body sits inside a scroll container so very long proteins
       (P300 = 2414 aa = ~16 wrapped blocks) are reachable by mouse
       wheel / PageDown rather than running off the bottom. */
    SequenceView { height: 1fr; }
    SequenceView #seq-scroll {
        height: 1fr;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    SequenceView _SequenceBody { height: auto; }
    SequenceView #seq-footer {
        height: 1;
        padding-right: 1;
        dock: bottom;
    }
    /* Scope to specific IDs so the Colorbar (a Static subclass) keeps its
       own width: 42 from DEFAULT_CSS rather than getting clobbered. */
    SequenceView #seq-ss-key,
    SequenceView #seq-domains-key { width: auto; padding-right: 2; }
    SequenceView #seq-spacer { width: 1fr; padding: 0; }
    """

    def __init__(self, project: BeakProject, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._sequence: str = ""
        self._plddt: Optional[np.ndarray] = None
        self._conservation: Optional[np.ndarray] = None
        self._sasa: Optional[np.ndarray] = None
        self._differential: Optional[np.ndarray] = None
        # Per-residue Pfam domain index (`-1` for unannotated positions).
        # Recomputed when domains are loaded; consumed by the "pfam"
        # color mode shared with the structure view.
        self._pfam_idx: Optional[np.ndarray] = None
        self._ss: str = ""
        self._domains: List[dict] = []
        self._status: str = "[dim]Loading sequence…[/dim]"
        # Match the persisted color mode so the colorbar + sequence
        # rendering line up with the dropdown on first paint.
        self._color_mode: str = (
            (project.manifest().get("view") or {}).get("color_mode", "plddt")
        )
        self._midpoint: float = 50.0
        # Domain bars and the SS track are both off on first project view —
        # they tile vertically below the sequence and crowd the layout
        # before the user has decided what they care about. Once a user
        # toggles either on (`d` / `s`), the choice is persisted in the
        # manifest's `[view]` block so future opens remember it.
        view_prefs = project.manifest().get("view") or {}
        self._show_domains: bool = bool(view_prefs.get("show_domains", False))
        self._show_ss: bool = bool(view_prefs.get("show_ss", False))

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="seq-scroll"):
            yield _SequenceBody(self, id="seq-body")
        with Horizontal(id="seq-footer"):
            # SS key pinned to the left; spacer pushes the rest right.
            yield Static("", id="seq-ss-key")
            yield Static("", id="seq-spacer")
            yield Static("", id="seq-domains-key")
            yield Colorbar(id="seq-colorbar")

    def on_mount(self) -> None:
        # Sync the inline colorbar to the persisted mode before any data
        # loads, so the gradient + label match the dropdown immediately.
        try:
            self.query_one("#seq-colorbar", Colorbar).set_mode(self._color_mode)
        except Exception:
            pass
        self._load()
        self._auto_pfam()

    def on_resize(self, event) -> None:
        if self._sequence:
            self._refresh_view()

    def reload(self) -> None:
        """Public hook for the screen to re-trigger a load.

        Resets the scroll container to the top — when reload is called
        on a set switch the meaning of "row N" changes (different
        wrapped block, different conservation overlay), so leaving the
        viewport mid-protein leaves the user looking at numerically the
        same residues with semantically different context.
        """
        try:
            scroll = self.query_one("#seq-scroll")
            scroll.scroll_home(animate=False)
        except Exception:
            pass
        self._load()

    def set_color_mode(self, mode: str) -> bool:
        if mode == "conservation" and self._conservation is None:
            return False
        if mode == "sasa" and self._sasa is None:
            return False
        if mode == "differential":
            try:
                from ..comparative import load_active_scores
                self._differential = load_active_scores(self._project)
            except Exception:
                self._differential = None
            if self._differential is None:
                return False
        if mode == "pfam" and (self._pfam_idx is None or not self._domains):
            return False
        self._color_mode = mode
        try:
            self.query_one("#seq-colorbar", Colorbar).set_mode(mode)
        except Exception:
            pass
        self._refresh_view()
        return True

    def set_midpoint(self, midpoint: float) -> None:
        self._midpoint = midpoint
        self._refresh_view()

    def _scalar_for_mode(self):
        if self._color_mode == "conservation" and self._conservation is not None:
            return self._conservation
        if self._color_mode == "sasa" and self._sasa is not None:
            return self._sasa
        if self._color_mode == "differential" and self._differential is not None:
            return self._differential
        if self._color_mode == "pfam" and self._pfam_idx is not None:
            return self._pfam_idx
        return self._plddt

    def on_colorbar_midpoint_changed(self, message: Colorbar.MidpointChanged) -> None:
        self._midpoint = message.midpoint
        self._refresh_view()
        # Let the screen forward this to the structure view too.

    def toggle_domains(self) -> bool:
        if not self._domains:
            return False
        self._show_domains = not self._show_domains
        try:
            with self._project.mutate() as m:
                m.setdefault("view", {})["show_domains"] = self._show_domains
        except Exception:
            pass
        self._refresh_view()
        return self._show_domains

    def has_domains(self) -> bool:
        return bool(self._domains)

    def has_pfam(self) -> bool:
        """True iff Pfam coloring would actually paint anything."""
        return self._pfam_idx is not None and bool(self._domains)

    def _compute_pfam_idx(self) -> Optional[np.ndarray]:
        """Build a per-residue array: domain index, or `-1` outside any.

        Domains are indexed in manifest order so the colors line up
        with the bars `_domain_line` already paints. Overlaps resolve
        to the first hit (consistent with how the bars draw).
        """
        if not self._sequence or not self._domains:
            return None
        n = len(self._sequence)
        idx = np.full(n, -1, dtype=np.int8)
        for i, d in enumerate(self._domains):
            try:
                start = max(0, int(d.get("env_from", 1)) - 1)
                end = min(n, int(d.get("env_to", 0)))
            except (TypeError, ValueError):
                continue
            for pos in range(start, end):
                if idx[pos] == -1:
                    idx[pos] = i
        return idx

    def _refresh_view(self) -> None:
        """Update the body Static and footer keys.

        `refresh(layout=True)` is load-bearing: toggling the domain bar
        or SS track changes how many lines `_render_body` emits per
        wrapped block, so the body's `height: auto` needs a fresh
        layout pass to pick up the new size. A plain `refresh()` only
        repaints inside the previously-computed geometry, which on the
        domains toggle clips the sequence rows out of the visible
        scroll area and the panel looks empty.
        """
        try:
            body = self.query_one("#seq-body", _SequenceBody)
            body.refresh(layout=True)
        except Exception:
            pass
        try:
            ss_legend = self._ss_legend() if (self._show_ss and self._ss) else ""
            self.query_one("#seq-ss-key", Static).update(ss_legend)
            cb = self.query_one("#seq-colorbar", Colorbar)
            cb.set_midpoint(self._midpoint)
            # Pfam coloring is categorical, so the gradient bar would
            # be misleading — hide it and let the domain chip legend
            # do the talking.
            cb.display = self._color_mode != "pfam"
            self.query_one("#seq-domains-key", Static).update(self._domains_legend())
        except Exception:
            pass

    def toggle_ss(self) -> bool:
        if not self._ss:
            return False
        self._show_ss = not self._show_ss
        try:
            with self._project.mutate() as m:
                m.setdefault("view", {})["show_ss"] = self._show_ss
        except Exception:
            pass
        self._refresh_view()
        return self._show_ss

    @work(thread=True, exclusive=True, group="auto-pfam")
    def _auto_pfam(self) -> None:
        """Run hmmscan once per project, cache to manifest. Idempotent."""
        manifest = self._project.manifest()
        if (manifest.get("domains") or {}).get("hits"):
            return

        try:
            from ...cli._common import get_hmmer_manager
            conn, hmmer = get_hmmer_manager()
            try:
                hits = hmmer.scan(str(self._project.target_sequence_path))
            finally:
                conn.close()
        except Exception:
            return

        from datetime import datetime
        with self._project.mutate() as manifest:
            manifest["domains"] = {
                "n_hits": len(hits),
                "last_updated": datetime.now(),
                "hits": [
                    {
                        "pfam_id": h.get("pfam_id", ""),
                        "pfam_name": h.get("pfam_name", ""),
                        "env_from": int(h.get("env_from", 0)),
                        "env_to": int(h.get("env_to", 0)),
                        "i_evalue": float(h.get("i_evalue", 1.0)),
                    }
                    for h in hits
                ],
            }
            self._domains = list(manifest["domains"]["hits"])
        self._pfam_idx = self._compute_pfam_idx()
        self.app.call_from_thread(self._refresh_view)

    @work(thread=True, exclusive=True, group="sequence-load")
    def _load(self) -> None:
        try:
            sequence = self._project.target_sequence()
        except Exception as e:  # noqa: BLE001
            self._status = f"[red]Sequence load error: {e}[/red]"
            self.app.call_from_thread(self._refresh_view)
            return

        plddt: Optional[np.ndarray] = None
        ss: str = ""

        target = self._project.manifest().get("target", {}) or {}
        uniprot = target.get("uniprot_id")
        if uniprot:
            cif_path = self._project.path / "structures" / f"{uniprot}_AF.cif"
            if cif_path.exists():
                try:
                    from ..structure import load_ca_coords, parse_secondary_structure
                    _, p = load_ca_coords(cif_path)
                    if len(p) == len(sequence):
                        plddt = p
                    ss = parse_secondary_structure(cif_path, len(sequence))
                except Exception:
                    pass

        manifest = self._project.manifest()
        domains = (manifest.get("domains") or {}).get("hits") or []

        # Quick conservation from hits.fasta if a search has landed.
        conservation = None
        try:
            from ..conservation import compute_quick_conservation
            conservation = compute_quick_conservation(self._project)
        except Exception:
            pass

        # SASA from the cached CIF if available.
        sasa = None
        if uniprot:
            cif_path = self._project.path / "structures" / f"{uniprot}_AF.cif"
            if cif_path.exists():
                try:
                    from ..structure import load_sasa
                    s = load_sasa(cif_path, len(sequence))
                    if s is not None and len(s) == len(sequence):
                        sasa = s
                except Exception:
                    pass

        self._sequence = sequence
        self._plddt = plddt
        self._conservation = conservation
        self._sasa = sasa
        self._ss = ss
        self._domains = list(domains)
        self._pfam_idx = self._compute_pfam_idx()
        self.app.call_from_thread(self._refresh_view)

    # ---- rendering (called by inner _SequenceBody.render) ----

    def _render_body(self, width: int) -> str:
        if not self._sequence:
            return self._status

        from ..structure import color_for_mode

        if width < 20:
            return ""
        block_w = max(20, width - 14)

        n = len(self._sequence)
        ss = self._ss if self._ss else ' ' * n

        lines = []
        for start in range(0, n, block_w):
            end = min(start + block_w, n)
            seq_chunk = self._sequence[start:end]
            ss_chunk = ss[start:end]

            ss_line = ''.join(
                f"[{_SS_COLORS[c]}]{c}[/{_SS_COLORS[c]}]" if c in _SS_COLORS else ' '
                for c in ss_chunk
            )

            scalar = self._scalar_for_mode()
            if scalar is not None:
                seq_parts = []
                for i, aa in enumerate(seq_chunk):
                    color = color_for_mode(scalar[start + i], self._color_mode, self._midpoint)
                    seq_parts.append(f"[{color}]{aa}[/{color}]")
                seq_line = ''.join(seq_parts)
            else:
                seq_line = seq_chunk

            pos_label = f"[dim]{start + 1:>5}[/dim]"
            end_label = f"[dim]{end:>5}[/dim]"

            if self._show_domains and self._domains:
                lines.append(f"      {self._domain_line(start, end)}")
            if self._show_ss and self._ss:
                lines.append(f"      {ss_line}")
            lines.append(f"{pos_label} {seq_line} {end_label}")
            lines.append("")

        # Legend lives in its own Static at the bottom of the panel — see
        # `#seq-legend` in compose() — so it stays anchored regardless of
        # sequence length.
        return "\n".join(lines).rstrip("\n")

    def _domain_line(self, start: int, end_exc: int) -> str:
        cells: list = [None] * (end_exc - start)
        for i, d in enumerate(self._domains):
            color = _DOMAIN_PALETTE[i % len(_DOMAIN_PALETTE)]
            ds = max(0, int(d.get("env_from", 1)) - 1 - start)
            de = min(end_exc - start, int(d.get("env_to", end_exc)) - start)
            if ds >= len(cells) or de <= 0:
                continue
            for j in range(max(0, ds), min(de, len(cells))):
                cells[j] = (color, d.get("pfam_name", ""))

        parts = []
        i = 0
        while i < len(cells):
            if cells[i] is None:
                j = i
                while j < len(cells) and cells[j] is None:
                    j += 1
                parts.append(" " * (j - i))
                i = j
            else:
                color, name = cells[i]
                j = i
                while j < len(cells) and cells[j] == (color, name):
                    j += 1
                width = j - i
                if width >= len(name) + 2 and name:
                    pad = (width - len(name)) // 2
                    seg = "─" * pad + name + "─" * (width - pad - len(name))
                else:
                    seg = "─" * width
                parts.append(f"[{color}]{seg}[/{color}]")
                i = j
        return "".join(parts)

    def _ss_legend(self) -> str:
        h = _SS_COLORS['H']
        e = _SS_COLORS['E']
        return (
            f"[{h}]H[/{h}] helix  "
            f"[{e}]E[/{e}] strand  "
            f"[dim]· loop[/dim]"
        )

    def _domains_legend(self) -> str:
        # Pfam color mode replaces the gradient colorbar with this
        # categorical legend, so render the chips even when the user
        # has hidden the inline domain bars.
        active_in_pfam = self._color_mode == "pfam" and bool(self._domains)
        if not active_in_pfam and not (self._show_domains and self._domains):
            return ""
        parts = ["[dim]│[/dim]"]
        for i, d in enumerate(self._domains):
            color = _DOMAIN_PALETTE[i % len(_DOMAIN_PALETTE)]
            parts.append(f"[{color}]{d.get('pfam_name', '')}[/{color}]")
        return "  ".join(parts)
