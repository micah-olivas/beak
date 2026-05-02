"""Full-screen taxonomy view: bars, growth-temp histogram, trait grid.

Toggles via `t` from the project detail screen, but only after taxonomy
has been annotated (i.e. ``homologs/<active set>/taxonomy.parquet``
exists). Press ``space`` to cycle the taxa-bar grouping level (phylum
↔ domain ↔ class).

Panes (top to bottom):
    1. Counts by taxon — horizontal bar chart with stable per-group colours
    2. Growth-temperature histogram — bins coloured along a cold→hot ramp
    3. Trait composition grid — for each high-coverage boolean / small
       categorical trait, a compact bar chart of value counts

The screen is intentionally one big text-rendered Static so adding
panes later is just appending another `_render_*` method to the
composition list.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from .alignment_view import _PHYLUM_PALETTE


# Domain palette — fixed three-way map (Bacteria / Archaea / Eukaryota
# / Viruses) with a fallback for anything else.
_DOMAIN_COLORS = {
    "Bacteria":   "#4D96FF",
    "Archaea":    "#A66DD4",
    "Eukaryota":  "#6BCF7F",
    "Viruses":    "#FF9F45",
}
_DOMAIN_FALLBACK = "#888888"


# 8-cell horizontal bar progression — combined with full blocks for the
# integer part this yields ~1/8-cell precision on the bar tip.
_BAR_GLYPHS = " ▏▎▍▌▋▊▉"

# Growth-temp palette anchored on the canonical microbiology class
# boundaries — optimum growth temperature ≤15 °C (psychrophile),
# 20–45 °C (mesophile), 45–80 °C (thermophile), >80 °C
# (hyperthermophile). Linear interp between stops.
_TEMP_STOPS = (
    (0.0,   (0x35, 0x4D, 0xC8)),  # extreme cold
    (15.0,  (0x4D, 0x96, 0xFF)),  # psychrophile / mesophile boundary
    (30.0,  (0x6B, 0xCF, 0x7F)),  # central mesophile (~human body temp)
    (45.0,  (0xFF, 0xD9, 0x3D)),  # mesophile / thermophile boundary
    (65.0,  (0xFF, 0x9F, 0x45)),  # central thermophile
    (80.0,  (0xFF, 0x6B, 0x6B)),  # thermophile / hyperthermophile boundary
    (105.0, (0xC8, 0x35, 0x35)),  # extreme hyperthermophile (~Pyrolobus)
)


# Canonical class boundaries from the microbiology literature.
# Used both for the gradient stops above and the class-tally summary
# beneath the histogram. References (consensus across 6 papers):
#   - ThermoBase (PLOS ONE 2022)
#   - Anaerobic Thermophiles (Front. Microbiol. 2014)
#   - Genomic/metabolic networks of thermophiles vs psychrophiles (2025)
#   - "Freezing thermophiles" (Microorganisms 2022)
#   - Cytoplasmic fluidity in psychrophiles (bioRxiv 2025)
#   - Extremophile review (Curr. Genom. 2020)
_TEMP_CLASSES = (
    ("psychrophile",     None,  15.0, "#4D96FF"),
    ("mesophile",        15.0,  45.0, "#6BCF7F"),
    ("thermophile",      45.0,  80.0, "#FF9F45"),
    ("hyperthermophile", 80.0,  None, "#FF6B6B"),
)


_LEVELS = ("phylum", "domain", "class")


def _temp_color(t: float) -> str:
    """Map a temperature (°C) to a hex colour along the cold→hot stops."""
    if t <= _TEMP_STOPS[0][0]:
        r, g, b = _TEMP_STOPS[0][1]
        return f"#{r:02X}{g:02X}{b:02X}"
    if t >= _TEMP_STOPS[-1][0]:
        r, g, b = _TEMP_STOPS[-1][1]
        return f"#{r:02X}{g:02X}{b:02X}"
    for (t0, c0), (t1, c1) in zip(_TEMP_STOPS, _TEMP_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02X}{g:02X}{b:02X}"
    r, g, b = _TEMP_STOPS[-1][1]
    return f"#{r:02X}{g:02X}{b:02X}"


def _bar_str(frac: float, width: int) -> str:
    """Render a horizontal bar with 1/8-cell tip precision."""
    width = max(1, width)
    cells_8 = int(round(max(0.0, min(1.0, frac)) * width * 8))
    full = cells_8 // 8
    partial = cells_8 % 8
    bar = "█" * full
    if partial > 0 and full < width:
        bar += _BAR_GLYPHS[partial]
    return bar.ljust(width)


class _TaxonomyBody(Static):
    ALLOW_SELECT = False

    def __init__(self, parent: "TaxonomyViewerScreen", **kwargs) -> None:
        super().__init__("[dim]Loading taxonomy…[/dim]", **kwargs)
        self._parent = parent

    def render(self):
        return self._parent._render_body(self.size.width, self.size.height)

    def on_resize(self, event) -> None:
        self.refresh()


class TaxonomyViewerScreen(Screen):
    """Multi-pane taxonomy view."""

    BINDINGS = [
        Binding("escape", "back",         "Back"),
        Binding("space",  "cycle_level",  "Group"),
        Binding("d",      "level_domain", "Domain", show=False),
        Binding("p",      "level_phylum", "Phylum", show=False),
        Binding("c",      "level_class",  "Class",  show=False),
    ]

    DEFAULT_CSS = """
    TaxonomyViewerScreen _TaxonomyBody { height: 1fr; padding: 1 2; overflow-y: auto; }
    """

    def __init__(self, project) -> None:
        super().__init__()
        self._project = project
        self._tax: Optional[pd.DataFrame] = None
        self._traits: Optional[pd.DataFrame] = None
        self._level: str = "phylum"
        self._missing: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield _TaxonomyBody(self, id="tax-body")
        yield Footer()

    def on_mount(self) -> None:
        self._load()
        self._update_title()

    # ---- data ----

    def _load(self) -> None:
        h_dir = self._project.active_homologs_dir()
        path = h_dir / "taxonomy.parquet"
        if not path.exists():
            self._missing = True
            return
        try:
            tax = pd.read_parquet(path)
        except Exception:
            self._missing = True
            return

        # Drop the target row if it leaked in.
        target_uniprot = (
            (self._project.manifest().get("target") or {}).get("uniprot_id")
        )
        if target_uniprot and "uniprot_id" in tax.columns:
            tax = tax[tax["uniprot_id"] != target_uniprot]
        if "phylum" in tax.columns and tax["phylum"].notna().sum() == 0:
            self._level = "domain"
        self._tax = tax

        # Optional traits join. Read defensively — traits is a "nice to
        # have" pane; a stale or missing parquet shouldn't block the rest.
        traits_path = h_dir / "traits.parquet"
        if traits_path.exists():
            try:
                self._traits = pd.read_parquet(traits_path)
            except Exception:
                self._traits = None

    def _color_for_level(self, label: str) -> str:
        if self._level == "domain":
            return _DOMAIN_COLORS.get(label, _DOMAIN_FALLBACK)
        if label == "(unresolved)":
            return _DOMAIN_FALLBACK
        return _PHYLUM_PALETTE[hash(label) % len(_PHYLUM_PALETTE)]

    # ---- pane: counts by taxon ----

    def _render_taxa_pane(self, width: int) -> str:
        s = self._tax[self._level].fillna("(unresolved)")
        counts = list(s.value_counts(sort=True, ascending=False).items())
        if not counts:
            return f"[dim]No values at the {self._level!r} level.[/dim]"

        n_total = sum(c for _, c in counts)
        biggest = counts[0][1]
        max_label = max(len(label) for label, _ in counts)
        label_w = min(max(8, max_label), 32)
        count_w = max(3, len(str(biggest)))
        # Bars are for visual comparison, not absolute scale — capping
        # at ~40 cells keeps wide-screen layouts from turning a single
        # taxon row into a 130-cell streak that drowns out the labels.
        avail = max(8, width - (label_w + count_w + 12))
        bar_w = min(40, avail)

        header = (
            f"[bold cyan]Counts by {self._level}[/bold cyan]  "
            f"[dim]({n_total} sequences · "
            f"{int(self._tax[self._level].notna().sum())}/{len(self._tax)} resolved · "
            f"space cycles level)[/dim]"
        )
        rows: List[str] = []
        for label, n in counts:
            label = label or "(unresolved)"
            color = self._color_for_level(label)
            disp = label if len(label) <= label_w else label[: label_w - 1] + "…"
            disp = disp.ljust(label_w)
            count_str = str(n).rjust(count_w)
            pct = (n / n_total) * 100.0 if n_total else 0.0
            bar = _bar_str(n / biggest if biggest else 0.0, bar_w)
            rows.append(
                f"[{color}]{disp}[/{color}]  "
                f"[bold]{count_str}[/bold] "
                f"[dim]{pct:4.1f}%[/dim]  "
                f"[{color}]{bar}[/{color}]"
            )
        return header + "\n" + "\n".join(rows)

    # ---- pane: growth-temperature histogram ----

    def _render_temp_pane(self, width: int) -> str:
        col = "growth_temp"
        if self._tax is None or col not in self._tax.columns:
            return ""
        temps = self._tax[col].dropna().astype(float).to_numpy()
        if temps.size == 0:
            return ""

        # Bin edges: fixed 5 °C bins from a sensible floor to the data's
        # ceiling. Anchored ranges read better than auto-binned ones for
        # growth temp, which has well-known regions (psychro / meso /
        # thermo / hyper).
        floor = max(-5.0, np.floor(temps.min() / 5.0) * 5.0)
        ceil_ = min(120.0, np.ceil(temps.max() / 5.0) * 5.0)
        if ceil_ <= floor:
            ceil_ = floor + 5.0
        edges = np.arange(floor, ceil_ + 5.0, 5.0)
        counts, _ = np.histogram(temps, bins=edges)
        if counts.sum() == 0:
            return ""

        biggest = int(counts.max())
        # Layout: " 25-30 °C  (n=12)  ███████"
        label_w = 11  # "  25-30 °C "
        count_w = max(4, len(f"n={biggest}"))
        avail = max(8, width - (label_w + count_w + 6))
        bar_w = min(40, avail)

        header = (
            f"[bold cyan]Growth temperature[/bold cyan]  "
            f"[dim]({len(temps)} annotated · "
            f"min {temps.min():.0f}, median {np.median(temps):.0f}, "
            f"max {temps.max():.0f} °C · 5 °C bins)[/dim]"
        )
        rows: List[str] = []
        for i, n in enumerate(counts):
            lo = edges[i]
            hi = edges[i + 1]
            mid = 0.5 * (lo + hi)
            color = _temp_color(mid)
            label = f"{int(lo):3d}-{int(hi):<3d} °C".rjust(label_w)
            count_str = f"n={n}".rjust(count_w)
            bar = _bar_str(n / biggest if biggest else 0.0, bar_w)
            rows.append(
                f"[dim]{label}[/dim]  "
                f"[bold]{count_str}[/bold]  "
                f"[{color}]{bar}[/{color}]"
            )

        # Class-tally line — counts each homolog into the canonical
        # psychrophile / mesophile / thermophile / hyperthermophile bins.
        class_counts = []
        for name, lo, hi, color in _TEMP_CLASSES:
            mask = np.ones_like(temps, dtype=bool)
            if lo is not None:
                mask &= (temps >= lo)
            if hi is not None:
                mask &= (temps < hi)
            class_counts.append((name, int(mask.sum()), color))
        tally = "  ".join(
            f"[{c}]{name}[/{c}] [bold]{n}[/bold]"
            for name, n, c in class_counts if n > 0
        )
        return header + "\n" + "\n".join(rows) + (
            f"\n[dim]Classes:[/dim]  {tally}" if tally else ""
        )

    # ---- pane: trait composition ----

    # Traits worth surfacing first. Listed in display order — the renderer
    # skips entries not present or with zero coverage.
    _TRAIT_SHOWLIST = (
        ("trait_gram_positive",       "Gram positive"),
        ("trait_aerotolerant",        "Aerotolerant"),
        ("trait_obligate_anaerobic",  "Obligate anaerobe"),
        ("trait_presence_of_motility", "Motile"),
        ("trait_sporulation",         "Sporulating"),
    )

    # A pair of categorical traits where multiple discrete values are
    # interesting — render as a small horizontal bar per value.
    _CATEGORICAL_TRAITS = ()  # populated dynamically when available

    def _render_traits_pane(self, width: int) -> str:
        if self._traits is None or self._traits.empty:
            return ""
        n_total = len(self._traits)
        # Filter to the shortlist that's actually populated.
        rows_to_show: List[Tuple[str, str, pd.Series]] = []
        for col, label in self._TRAIT_SHOWLIST:
            if col not in self._traits.columns:
                continue
            s = self._traits[col].dropna()
            if s.empty:
                continue
            rows_to_show.append((col, label, s))
        if not rows_to_show:
            return ""

        header = (
            f"[bold cyan]Trait composition[/bold cyan]  "
            f"[dim]({n_total} sequences total · counts of true/false per trait · "
            f"NA = not in metaTraits)[/dim]"
        )
        # Compact bar pair per trait: one row per trait, true|false bars
        # share a single 0..n_known scale so traits are comparable at a
        # glance. Bar width is capped well below the panel width so the
        # row stays compact on wide terminals.
        label_w = max(len(label) for _, label, _ in rows_to_show) + 1
        avail = max(20, width - (label_w + 22))
        bar_w = min(40, avail)
        out = [header]
        for col, label, s in rows_to_show:
            # Coerce mixed object/bool dtypes through a single
            # str-lowered path. The previous version added the bool-
            # comparison count on top of the string-comparison count,
            # which double-counted any row where the underlying value
            # was a real Python `True` and produced negative `false`
            # counts (e.g. "92 ✓ -45 ✗" on an 81-row trait table).
            text = s.dropna().astype(str).str.strip().str.lower()
            n_true = int((text == "true").sum())
            n_false = int((text == "false").sum())
            n_known = n_true + n_false
            if n_known == 0:
                continue
            frac_true = n_true / n_known
            true_w = int(round(bar_w * frac_true))
            false_w = bar_w - true_w
            true_bar = "█" * true_w
            false_bar = "█" * false_w
            out.append(
                f"[bold]{label.ljust(label_w)}[/bold] "
                f"[#6BCF7F]{str(n_true).rjust(4)} ✓[/#6BCF7F] "
                f"[#FF6B6B]{str(n_false).rjust(4)} ✗[/#FF6B6B]  "
                f"[#6BCF7F]{true_bar}[/#6BCF7F][#FF6B6B]{false_bar}[/#FF6B6B]"
                f"  [dim]{n_known}/{n_total}[/dim]"
            )
        return "\n".join(out)

    # ---- body composition ----

    def _render_body(self, width: int, height: int) -> str:
        if self._missing:
            return (
                "[red]Taxonomy not yet annotated for this set.[/red]\n\n"
                "[dim]Pull homologs first; the taxonomy join runs "
                "automatically afterwards.[/dim]"
            )
        if self._tax is None or self._tax.empty:
            return "[dim]Loading taxonomy…[/dim]"

        # Each entry returns "" when its data isn't available; skip those
        # so the body stays tight.
        panes: List[Callable[[int], str]] = [
            self._render_taxa_pane,
            self._render_temp_pane,
            self._render_traits_pane,
        ]
        chunks = [pane(width) for pane in panes]
        chunks = [c for c in chunks if c]
        return "\n\n".join(chunks)

    # ---- title + bindings ----

    def _update_title(self) -> None:
        self.app.title = "beak"
        n = 0 if self._tax is None else len(self._tax)
        self.app.sub_title = (
            f"taxonomy · {self._project.name} ({n})  ·  level: {self._level}"
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_cycle_level(self) -> None:
        i = _LEVELS.index(self._level) if self._level in _LEVELS else 0
        self._level = _LEVELS[(i + 1) % len(_LEVELS)]
        self._refresh()

    def action_level_domain(self) -> None:
        self._level = "domain"
        self._refresh()

    def action_level_phylum(self) -> None:
        self._level = "phylum"
        self._refresh()

    def action_level_class(self) -> None:
        self._level = "class"
        self._refresh()

    def _refresh(self) -> None:
        self._update_title()
        try:
            self.query_one("#tax-body", _TaxonomyBody).refresh()
        except Exception:
            pass


def has_taxonomy(project) -> bool:
    """Cheap stat-check used by the detail screen to gate the binding."""
    return (project.active_homologs_dir() / "taxonomy.parquet").exists()
