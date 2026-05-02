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
    "differential": (
        "-1,#3535FF:0,#555555:1,#FF3535",
        (-1.0, 1.0),
        0.0,  # sentinel → neutral grey
    ),
    "plddt": (
        # Matches the AlphaFold confidence bands in tui/structure.py.
        "0,#FF7D45:50,#FFDB13:70,#65CBF3:90,#0053D6",
        (0.0, 100.0),
        0.0,
    ),
    "conservation": (
        "0,#FFFFFF:50,#FF8888:100,#CC0000",
        (0.0, 100.0),
        0.0,
    ),
    "sasa": (
        "0,#2E86AB:75,#888888:150,#FF7D45",
        (0.0, 150.0),
        75.0,
    ),
}


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


def _build_cxc_script(
    cif_filename: str,
    chain_name: str,
    palette: str,
    score_range: Tuple[float, float],
    title: Optional[str],
) -> str:
    lo, hi = score_range
    title_line = f"# {title}\n" if title else ""
    # Color key — ChimeraX `key` re-uses the same palette syntax as
    # `color`. Pinned to the lower-middle of the viewport so it doesn't
    # overlap the ribbon. tickLength + numericLabelSpacing keep the bar
    # readable; size is in fractional viewport units (width × height).
    return (
        f"{title_line}"
        f"open {cif_filename}\n"
        f"cartoon\n"
        f"# Per-residue scores live in the B-factor column; map them to colour.\n"
        f"color bfactor /{chain_name} palette {palette} range {lo:g},{hi:g}\n"
        f"# Defaults: white background, gentle ambient lighting, "
        f"silhouettes for crisp ribbon edges.\n"
        f"set bgColor white\n"
        f"lighting gentle\n"
        f"graphics silhouettes true\n"
        f"# Colour key (legend) along the bottom of the viewport.\n"
        f"key {palette} "
        f"ticks true tickLength 0.005 "
        f"numericLabelSpacing equal "
        f"pos 0.3,0.05 size 0.4,0.025 "
        f"showTool false\n"
    )


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

    cxc_out.write_text(_build_cxc_script(cif_out.name, chain_name, pal, rng, title))
    return cif_out, cxc_out, n_set
