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
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from ...project import BeakProject
from .colorbar import Colorbar


_SS_COLORS = {
    'H': '#FF6B6B',
    'E': '#FFD93D',
}

# Cycle through these for sequential Pfam domains.
_DOMAIN_PALETTE = ["#FF6B9D", "#FFA62B", "#A66DD4", "#3DCFD4", "#7DD87D"]


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
    SequenceView { height: auto; }
    SequenceView _SequenceBody { height: 1fr; }
    SequenceView #seq-footer {
        height: 1;
        padding-right: 1;
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
        self._ss: str = ""
        self._domains: List[dict] = []
        self._status: str = "[dim]Loading sequence…[/dim]"
        # Match the persisted color mode so the colorbar + sequence
        # rendering line up with the dropdown on first paint.
        self._color_mode: str = (
            (project.manifest().get("view") or {}).get("color_mode", "plddt")
        )
        self._midpoint: float = 50.0
        self._show_domains: bool = True
        self._show_ss: bool = bool(
            (project.manifest().get("view") or {}).get("show_ss", False)
        )

    def compose(self) -> ComposeResult:
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
        """Public hook for the screen to re-trigger a load."""
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
        return self._plddt

    def on_colorbar_midpoint_changed(self, message: Colorbar.MidpointChanged) -> None:
        self._midpoint = message.midpoint
        self._refresh_view()
        # Let the screen forward this to the structure view too.

    def toggle_domains(self) -> bool:
        if not self._domains:
            return False
        self._show_domains = not self._show_domains
        self._refresh_view()
        return self._show_domains

    def has_domains(self) -> bool:
        return bool(self._domains)

    def _refresh_view(self) -> None:
        """Update the body Static and footer keys."""
        try:
            body = self.query_one("#seq-body", _SequenceBody)
            body.refresh()
        except Exception:
            pass
        try:
            ss_legend = self._ss_legend() if (self._show_ss and self._ss) else ""
            self.query_one("#seq-ss-key", Static).update(ss_legend)
            self.query_one("#seq-colorbar", Colorbar).set_midpoint(self._midpoint)
            self.query_one("#seq-domains-key", Static).update(self._domains_legend())
        except Exception:
            pass

    def toggle_ss(self) -> bool:
        if not self._ss:
            return False
        self._show_ss = not self._show_ss
        try:
            m = self._project.manifest()
            m.setdefault("view", {})["show_ss"] = self._show_ss
            self._project.write(m)
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
        manifest = self._project.manifest()
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
        self._project.write(manifest)
        self._domains = manifest["domains"]["hits"]
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
        if not (self._show_domains and self._domains):
            return ""
        parts = ["[dim]│[/dim]"]
        for i, d in enumerate(self._domains):
            color = _DOMAIN_PALETTE[i % len(_DOMAIN_PALETTE)]
            parts.append(f"[{color}]{d.get('pfam_name', '')}[/{color}]")
        return "  ".join(parts)
