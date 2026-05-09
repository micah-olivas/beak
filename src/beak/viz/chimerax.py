"""Export a coloured structure for ChimeraX.

Writes two files into ``out_dir``:
    1. ``<name>.cif`` — the original mmCIF with per-residue scores written
       into the B_iso_or_equiv column. Residues with no score are left at
       a sentinel B-factor that the ChimeraX script renders as neutral.
    2. ``<name>.cxc`` — a ChimeraX command script that opens the CIF and
       applies a palette appropriate for the score type (diverging for
       differential / signed scores, sequential for pLDDT / conservation).

Usage in ChimeraX:
    open <name>.cxc

The user can then edit the .cxc to tweak rendering without re-running
beak. Scripts are intentionally short and human-readable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import gemmi
import numpy as np


# Common install locations for ChimeraX, checked in order. The first hit
# is invoked with the .cxc as its sole argument; ChimeraX treats .cxc
# files as command scripts.
_CHIMERAX_CANDIDATES_MAC = (
    "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
    "/Applications/ChimeraX-daily.app/Contents/MacOS/ChimeraX",
)
_CHIMERAX_CANDIDATES_LINUX = (
    "/usr/bin/chimerax",
    "/usr/local/bin/chimerax",
    "/opt/UCSF/ChimeraX/bin/ChimeraX",
)


def _resolve_chimerax_binary() -> Optional[str]:
    """Locate a ChimeraX binary on this machine. None if not found."""
    # $CHIMERAX takes precedence so users with custom installs can
    # override without editing this file.
    env = os.environ.get("CHIMERAX")
    if env and Path(env).exists():
        return env
    for name in ("chimerax", "ChimeraX"):
        found = shutil.which(name)
        if found:
            return found
    candidates = (
        _CHIMERAX_CANDIDATES_MAC if sys.platform == "darwin"
        else _CHIMERAX_CANDIDATES_LINUX
    )
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def open_in_chimerax(cxc_path: Path) -> Optional[str]:
    """Launch ChimeraX on a .cxc file. Non-blocking.

    Returns the binary used (informational), or None if ChimeraX wasn't
    found and the system fallback was invoked instead. Raises only if
    *both* the direct binary and the fallback fail.
    """
    cxc_path = Path(cxc_path)
    binary = _resolve_chimerax_binary()
    if binary is not None:
        # Run from the file's directory so the relative `open <name>.cif`
        # in the script resolves against the export folder.
        subprocess.Popen(
            [binary, str(cxc_path)],
            cwd=str(cxc_path.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return binary
    # Fall back to the OS handler (registered .cxc association on macOS,
    # xdg-open on Linux). May silently no-op if ChimeraX isn't installed.
    if sys.platform == "darwin":
        subprocess.Popen(
            ["open", str(cxc_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return None
    if sys.platform.startswith("linux"):
        subprocess.Popen(
            ["xdg-open", str(cxc_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return None
    raise RuntimeError("ChimeraX binary not found and no system fallback")


# Per-mode default palettes + ranges. ChimeraX's `color bfactor` accepts
# a comma-separated list of value:hex stops. The first stop's value is
# used as the "no-data" sentinel for residues outside the target
# sequence — the palette already paints it neutrally so we don't need a
# second pass for unmapped residues.
_PALETTES = {
    # Always use the **value:color** palette form (e.g.
    # `0,lightgray:100,red`) — never the bare-color shorthand
    # (`lightgray:red`). ChimeraX's `key` command derives its numeric
    # labels from the values embedded in the palette; the shorthand
    # form leaves the labels blank (or defaulted to 0..1) so the
    # legend displays without any numeric scale. The trade-off vs.
    # the cleaner-looking shorthand is worth it for a readable bar.
    "differential": (
        # Diverging blue → neutral grey → red. Gray landing at 0 makes
        # the centre of the [-1, 1] range neutral.
        "-1,blue:0,gray:1,red",
        (-1.0, 1.0),
        0.0,  # sentinel → neutral grey
    ),
    "plddt": (
        # AlphaFold confidence bands at 50 / 70 / 90 — value:color
        # stops so the orange/yellow/cyan/blue boundaries land
        # exactly where the AF tool expects them.
        "0,orange:50,yellow:70,cyan:90,blue",
        (0.0, 100.0),
        0.0,
    ),
    "bfactor": (
        # PDB B-factor / temperature factor (Å²). Standard "spectrum b"
        # thermal coloring used by PyMOL / ChimeraX: low B (rigid /
        # ordered) is cool blue, high B (mobile / disordered) is hot
        # red, neutral mid stays white. Range 0–50 Å² covers the
        # typical band for well-resolved crystal structures (high-res
        # backbones cluster 5–25, surface side chains 30–60); the
        # palette saturates above 50 so disordered loops don't wash
        # out the rest of the structure. Sentinel = 0 so unmapped
        # residues read as the most-ordered color rather than as
        # "missing data" red.
        "0,blue:25,white:50,red",
        (0.0, 50.0),
        0.0,
    ),
    "conservation": (
        # 3-stop gradient with a midpoint at 50 — without it, the
        # linear 0→100 lightgray-to-red ramp puts typical alignment
        # conservation values (most positions sit in the 20–50 band)
        # into the visually-flat lightgray end of the bar, and the
        # whole structure reads as grey. Anchoring 50 → medium pink
        # pushes the visible color into the biologically-interesting
        # range; 100 stays a deep red so the most-conserved positions
        # still pop. Mirrors the TUI's `conservation_color()` shape
        # at its default midpoint=50.
        "0,lightgray:50,#FF8888:100,#7F0000",
        (0.0, 100.0),
        0.0,
    ),
    "sasa": (
        # **Relative SASA (%)** — Tien et al. 2013 normalisation: the
        # observed SASA divided by the per-residue maximum from the
        # extended Gly-X-Gly tripeptide table. The literature
        # consensus is to classify burial / exposure off rSASA, not
        # absolute Å², because the absolute scale spans 85 Å²
        # (Gly max) to 259 Å² (Trp max) — a single absolute threshold
        # misclassifies small vs. large residues by construction.
        # Stops anchored to the standard cutoffs (Tsishyn 2025;
        # Singh & Ahmad 2009; Tien 2013):
        #   <  20 %  → buried             (dodgerblue → light blue)
        #   20-50 %  → partially exposed  (light blue → white)
        #   ≥  50 %  → exposed            (white → orange → red)
        # Sentinel = mid-of-partial so unmapped residues read as
        # neutral rather than as either buried or exposed by default.
        "0,dodgerblue:20,lightskyblue:50,white:75,orange:100,#CC0000",
        (0.0, 100.0),
        35.0,
    ),
}


# Mode-specific score thresholds for the "highlight as sticks" overlay.
# A residue earns a stick if its score crosses this threshold (sign for
# differential — symmetric around 0). Tuned so a typical project
# surfaces ~10-25% of residues as sticks: dense enough that the
# highlighted side chains read as a pattern rather than scattered
# outliers, sparse enough that the cartoon underneath stays readable.
#
# Bumped conservation down from "top N by score" (which surfaced
# 5-15 residues regardless of distribution) to a flat 60% threshold —
# captures well-conserved positions including non-catalytic anchor
# residues, not just the absolute peaks.
_HIGHLIGHT_THRESHOLDS = {
    "conservation": 60.0,        # ≥ 60% conserved
    "plddt": 90.0,               # AlphaFold's "high confidence" band
    # B-factor highlight is the *flexible* end of the scale (high B
    # = disordered / mobile residues). For well-resolved crystals,
    # B ≥ 40 Å² typically marks loop / surface flexibility worth
    # calling out. Skipping ≥ 50 here because at the palette's red
    # ceiling the structure context is already lost.
    "bfactor": 40.0,
    "sasa": 50.0,                # rSASA ≥ 50 % → exposed (Tien 2013 convention)
    "differential": 0.2,         # |score| ≥ 0.2
}

# Hard cap on stick count even when the threshold catches more — past
# ~30 the side-chain forest occludes the cartoon and the highlight
# stops carrying signal.
_HIGHLIGHT_MAX = 30


def _scores_to_residue_map(
    scores: np.ndarray,
    target_seq: str,
    cif_path: Path,
    chain_id: Optional[str],
) -> Tuple[Optional[str], dict]:
    """Map ``scores`` (indexed by target position) → ``{resnum: score}``.

    Returns ``(chain_name, residue_score_map)``. Uses the project's
    existing target → structure aligner so insertion codes and chain
    breaks are handled the same way the structure-view already does.
    """
    from ..structures.mapping import map_target_to_structure
    df = map_target_to_structure(target_seq, str(cif_path), chain_id)
    if df.empty:
        return None, {}
    out: dict = {}
    n = len(scores)
    for _, row in df.iterrows():
        i = int(row["target_pos"]) - 1
        if 0 <= i < n:
            out[int(row["pdb_resnum"])] = float(scores[i])
    # Resolve which chain we ended up on so the .cxc script can target it.
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    chain_name = chain_id
    if chain_name is None:
        for c in model:
            if len(c.get_polymer()) > 0:
                chain_name = c.name
                break
    return chain_name, out


def _write_cif_with_bfactors(
    src_cif: Path,
    dst_cif: Path,
    chain_name: str,
    score_by_resnum: dict,
    sentinel: float,
) -> int:
    """Write a CIF copy with B_iso_or_equiv replaced. Returns # residues set."""
    structure = gemmi.read_structure(str(src_cif))
    n_set = 0
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for residue in chain:
                score = score_by_resnum.get(residue.seqid.num)
                bf = float(score) if score is not None else sentinel
                # Set on every atom — `color bfactor` reads from atoms.
                for atom in residue:
                    atom.b_iso = bf
                if score is not None:
                    n_set += 1
    structure.make_mmcif_document().write_file(str(dst_cif))
    return n_set


_AXIS_TITLES = {
    "differential": "Differential score (low → high)",
    # Two distinct physical quantities, both stored in the cif's B-iso
    # column but with completely different meaning and units. Beak
    # picks the right title (and palette + range below) at export
    # time based on whether the loaded structure is AlphaFold or PDB.
    "plddt": "AlphaFold pLDDT",
    "bfactor": "B-factor (Å²)",
    "conservation": "Conservation (Jensen-Shannon, %)",
    # Relative SASA = absolute / Tien 2013 max-per-residue. % units; the
    # "buried" / "exposed" thresholds (20 / 50) come from the rSASA
    # literature convention (see structures/features.py).
    "sasa": "Relative SASA (%)",
}

# Compact units string rendered as a smaller second line under the
# colour-key title in ChimeraX exports. Empty string (or missing key)
# suppresses the units row entirely.
_AXIS_UNITS = {
    "conservation": "SJ divergence",
    "differential": "signed SJ divergence",
    "plddt": "0–100",
    "bfactor": "Å²",
    "sasa": "% rel. surface area",
}


def _palette_to_key_form(palette: str) -> str:
    """Convert a `color`-style palette to ChimeraX's `key`-style palette.

    `color byattribute … palette …` and `key …` accept different
    palette syntaxes:
      * `color` form:  `value1,color1:value2,color2:…`  (this module's
        canonical representation)
      * `key`   form:  `color1:value1 color2:value2 …`  (space-separated
        stops; values come AFTER the color)
    Without the conversion, the `key` command silently drops the
    numeric labels because it can't parse the values out of the
    `value,color:…` form — that's the bug the user saw as "no tick
    labels on the cbar".
    """
    stops = palette.split(":")
    converted = []
    for stop in stops:
        # Each stop is `<value>,<color>`. Split only on the first comma
        # because color names never contain commas but rgb tuples (if
        # we ever use them) do.
        if "," not in stop:
            # Shouldn't happen with our palette table, but if it does
            # the safe move is to leave the stop as-is and let
            # ChimeraX complain rather than fabricating a value.
            converted.append(stop)
            continue
        value, color = stop.split(",", 1)
        converted.append(f"{color}:{value}")
    return " ".join(converted)


def _build_cxc_script(cif_filename: str, helper_filename: str) -> str:
    """Minimal `.cxc` that opens the cif and hands off to the helper `.py`.

    The bulk of the rendering logic lives in the sibling Python script
    instead of inline ChimeraX commands so we can capture the
    just-opened model's auto-assigned id at runtime (`session.models
    .list()[-1].id`) and scope every coloring command to that single
    model. Doing this with cxc-only commands isn't possible in
    ChimeraX 1.8: there's no `python` inline command (added in 1.9+),
    `open … id N` silently drops large ids, and atom-spec syntax
    can't reference a model by name. `runscript` IS in 1.8 and
    exposes `session`, so it's the portable path that works on the
    user's current install without forcing an upgrade.
    """
    return (
        f"# Auto-generated by beak. Opens the cif and hands off to the\n"
        f"# sibling .py for model-aware coloring. The split is deliberate:\n"
        f"# the .py captures the just-opened model's auto-assigned id at\n"
        f"# runtime so the coloring commands don't bleed across other\n"
        f"# beak exports the user might already have loaded.\n"
        f"open {cif_filename}\n"
        f"runscript {helper_filename}\n"
    )


def _build_helper_script(
    chain_name: str,
    palette: str,
    score_range: Tuple[float, float],
    title: Optional[str],
    mode: str,
    highlight_resnums: Iterable[int] = (),
) -> str:
    """Python helper invoked by the `.cxc` via `runscript`.

    `runscript` injects the active ChimeraX `session` as a module
    global, so we can introspect the just-opened model and feed its
    actual id into every subsequent ChimeraX command. Every command
    here is also a valid ChimeraX command-line command — only the
    "which model are we operating on?" piece needs the runtime
    Python.
    """
    lo, hi = score_range
    axis_title = title or _AXIS_TITLES.get(mode, mode)
    # Single-quote-safe — `axis_title` lands inside Python single
    # quotes that wrap a ChimeraX `2dlabels` argument that itself
    # ends up inside ChimeraX double-quotes. Backslash-escape any
    # nested double quotes.
    axis_title_safe = axis_title.replace('\\', '\\\\').replace('"', '\\"')
    axis_units = _AXIS_UNITS.get(mode, "")
    axis_units_safe = axis_units.replace('\\', '\\\\').replace('"', '\\"')

    resnums = list(highlight_resnums)
    resnum_spec = ",".join(str(r) for r in resnums) if resnums else ""

    # `key` palette form differs from `color byattribute palette` —
    # see _palette_to_key_form for the rationale.
    key_palette = _palette_to_key_form(palette)

    return f'''# Auto-generated by beak. Run via `runscript <name>.py` from the
# sibling .cxc. `session` is provided by ChimeraX's runscript host.
#
# This script only runs the *dynamic* parts of the export: identify
# which model the .cxc just opened, then dispatch a fixed sequence of
# ChimeraX commands scoped to that model. Doing it from Python lets
# us address the just-opened model by its actual auto-assigned id
# instead of guessing a hardcoded `#1` (which would be wrong whenever
# the user has other structures loaded, and was the root cause of
# beak exports recoloring previously-loaded structures).

# `chimerax.core.commands.run` is the supported entry point for
# issuing command-line commands from Python — `session.commands.run`
# isn't an attribute on the Session object, that was a wrong API guess.
from chimerax.core.commands import run as _cx_run

def _run(cmd):
    _cx_run(session, cmd)

# Most-recently-opened model. `runscript` runs after the .cxc's
# `open`, so this is the structure beak just wrote.
_m = session.models.list()[-1]
_mid = "#" + ".".join(str(p) for p in _m.id)

_run(f"cartoon {{_mid}}")

# Per-residue scores live in the B-factor column. `color byattribute
# bfactor` (NOT the Chimera-era `color bfactor` alias) is the
# canonical ChimeraX command. `target abcs` paints atoms / bonds /
# cartoon / surfaces; `_mid & protein` scopes to this export's
# protein atoms only.
_run(
    f"color byattribute bfactor {{_mid}} & protein "
    f"palette {palette} range {lo:g},{hi:g} target abcs"
)
# Recolor heteroatoms by element on top of the bfactor coloring —
# carbons keep their bfactor color, N/O/S get standard element
# colors so any side-chain sticks read clearly.
_run(f"color {{_mid}} & protein byhetero")

_run("set bgColor white")
_run("lighting gentle")
_run("graphics silhouettes true")
''' + (
        f'''
# Top {len(resnums)} residue(s) by {mode} score — show as sticks so
# the highlighted side chains stand out against the cartoon.
_run(f"show {{_mid}}/{chain_name}:{resnum_spec} atoms")
_run(f"style {{_mid}}/{chain_name}:{resnum_spec} stick")
'''
        if resnums else ""
    ) + f'''
# Colour key (legend). `key`'s palette syntax is space-separated
# `color:value` (different from `color byattribute palette`'s
# `value,color:…` form), so we use the converted form here.
# `fontSize` + `labelColor` make the numeric tick labels readable
# at default zoom; the labels themselves come from the values
# embedded in the palette stops.
_run(
    f"key {key_palette} "
    f"ticks true tickLength 8 "
    f"fontSize 18 labelColor black "
    f"numericLabelSpacing equal "
    f"pos 0.3,0.07 size 0.4,0.025 "
    f"showTool false"
)
# Title + units labels — fixed names (`beak_keytitle`, `beak_keyunits`)
# so re-running another export *replaces* the previous labels rather
# than stacking new text on top at the same xpos/ypos. The first
# export's create succeeds; every subsequent export hits the existing
# label and raises `UserError: Label "..." already exists`, which we
# catch and fall back to `2dlabels change` (the in-place update
# subcommand). Mirrors the singleton behaviour of the `key` command
# itself: the active legend + title always reflect the most-recently-
# loaded export.
#
# Layout: title centered at xpos 0.5 above the bar; units a smaller
# second line below the title (still above the bar). xpos 0.5 with
# default-centered alignment keeps the text visually centred on the
# bar (which spans 0.3..0.7 with size 0.4 from the `key` command).
_title_args = (
    f'text "{axis_title_safe}" '
    f'xpos 0.5 ypos 0.135 color black size 16 '
    f'visibility true'
)
try:
    _run(f"2dlabels create beak_keytitle {{_title_args}}")
except Exception:
    _run(f"2dlabels change beak_keytitle {{_title_args}}")
# Units row — only shown when the mode has a units string defined
# in `_AXIS_UNITS`. We always issue the command (with empty string
# when no units) so a second export that switches modes correctly
# clears any previously-shown units string.
_units_args = (
    f'text "{("(" + axis_units_safe + ")") if axis_units_safe else ""}" '
    f'xpos 0.5 ypos 0.108 color #555555 size 11 '
    f'visibility true'
)
try:
    _run(f"2dlabels create beak_keyunits {{_units_args}}")
except Exception:
    _run(f"2dlabels change beak_keyunits {{_units_args}}")
'''


def _pick_highlight_resnums(
    score_by_resnum: dict, mode: str, threshold: Optional[float] = None,
) -> list:
    """Pick the residues to render as sticks for the active mode.

    Threshold-based: a residue earns a stick when its score crosses
    the mode's threshold (``_HIGHLIGHT_THRESHOLDS``). Differential is
    treated as a magnitude — interesting residues sit at *both* ends
    of a diverging scale, so the criterion is ``|score| ≥ threshold``.
    Pfam is categorical (the per-residue "score" is a domain index),
    so highlighting would just pick arbitrary residues from the
    longest domain — skip it.

    The hard cap (``_HIGHLIGHT_MAX``) keeps the side-chain density
    readable on densely-conserved proteins; the cap is applied after
    sorting by score (descending — most extreme first) so we keep the
    highest-signal residues when more cross the threshold than fit.

    Pass an explicit ``threshold`` to override the per-mode default.
    """
    if not score_by_resnum or mode == "pfam":
        return []
    if threshold is None:
        threshold = _HIGHLIGHT_THRESHOLDS.get(mode)
        if threshold is None:
            return []  # mode without a sensible default; caller can opt in

    if mode == "differential":
        passing = [
            (rn, s) for rn, s in score_by_resnum.items() if abs(s) >= threshold
        ]
        passing.sort(key=lambda rs: abs(rs[1]), reverse=True)
    else:
        passing = [
            (rn, s) for rn, s in score_by_resnum.items() if s >= threshold
        ]
        passing.sort(key=lambda rs: rs[1], reverse=True)
    return [rn for rn, _ in passing[:_HIGHLIGHT_MAX]]


def export_chimerax(
    cif_path: Path,
    scores: Iterable[float],
    target_seq: str,
    out_dir: Path,
    name: str,
    mode: str = "differential",
    chain_id: Optional[str] = None,
    title: Optional[str] = None,
    palette: Optional[str] = None,
    score_range: Optional[Tuple[float, float]] = None,
    highlight_threshold: Optional[float] = None,
) -> Tuple[Path, Path, int]:
    """Write a coloured CIF + matching ChimeraX command script.

    Args:
        cif_path: Source mmCIF file (typically the AlphaFold prediction
            already cached in the project).
        scores: Per-target-residue numeric scores. Length must equal
            ``len(target_seq)``.
        target_seq: Reference (target) amino acid sequence.
        out_dir: Directory to write outputs into. Created if absent.
        name: Basename for the CIF + cxc files.
        mode: One of "differential", "plddt", "conservation", "sasa".
            Selects the default palette + range. Override via ``palette``
            and ``score_range`` if needed.
        chain_id: Force a specific chain. Default picks the first
            polymer chain.
        title: Human-readable label for the script's leading comment
            (e.g. ``"temperature_growth ≥ 50"``).
        palette, score_range: Optional explicit overrides.

    Returns:
        ``(cif_out, cxc_out, n_residues_set)``. ``n_residues_set`` is the
        number of structure residues that received a real score (vs. the
        sentinel for unmapped positions).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path = Path(cif_path)

    arr = np.asarray(list(scores), dtype=np.float64)
    if len(arr) != len(target_seq):
        raise ValueError(
            f"scores length {len(arr)} != target sequence length {len(target_seq)}"
        )

    chain_name, residue_scores = _scores_to_residue_map(
        arr, target_seq, cif_path, chain_id
    )
    if chain_name is None:
        raise ValueError(f"No polymer chain found in {cif_path}")

    pal, rng, sentinel = _PALETTES.get(mode, _PALETTES["differential"])
    if palette is not None:
        pal = palette
    if score_range is not None:
        rng = score_range

    cif_out = out_dir / f"{name}.cif"
    cxc_out = out_dir / f"{name}.cxc"
    n_set = _write_cif_with_bfactors(
        cif_path, cif_out, chain_name, residue_scores, sentinel
    )

    highlight_resnums = _pick_highlight_resnums(
        residue_scores, mode, highlight_threshold,
    )

    # Sibling `.py` carries the dynamic, model-aware coloring (capturing
    # the just-opened model's id from `session.models`). The `.cxc` is
    # minimal — opens the cif and dispatches to the helper via
    # `runscript`. This split is what lets a beak export render
    # correctly even when the user has unrelated models open in the
    # same ChimeraX window.
    py_out = out_dir / f"{name}.py"
    py_out.write_text(_build_helper_script(
        chain_name, pal, rng, title, mode, highlight_resnums,
    ))
    cxc_out.write_text(_build_cxc_script(cif_out.name, py_out.name))
    return cif_out, cxc_out, n_set
