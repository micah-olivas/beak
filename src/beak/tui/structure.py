"""Structure loading + braille rendering for the TUI.

Uses gemmi (already a dep) to read mmCIF and pull Cα coordinates +
B-factor (pLDDT for AlphaFold predictions). Projects to 2D and
renders to a braille-character grid colored by AlphaFold pLDDT bands.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# AlphaFold pLDDT color bands.
# Order matters: first matching threshold wins (descending).
_PLDDT_BANDS = [
    (90, "#0053D6"),   # very high (blue)
    (70, "#65CBF3"),   # confident (cyan)
    (50, "#FFDB13"),   # low (yellow)
    (0,  "#FF7D45"),   # very low (orange)
]

# Braille cell layout: 2 cols x 4 rows of dots per character cell.
# Maps (col_in_cell, row_in_cell) -> bit index in U+2800 + mask.
# https://en.wikipedia.org/wiki/Braille_Patterns
_BRAILLE_BITS = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 6,
    (1, 0): 3, (1, 1): 4, (1, 2): 5, (1, 3): 7,
}


def read_cif_meta(cif_path: Path) -> dict:
    """Return resolution and experimental method extracted from a CIF header.

    Keys:
      ``resolution`` — float (Å) or None; populated for X-ray and EM
      ``method``     — abbreviated string or None ("X-ray", "cryo-EM", "NMR", …)

    Returns an empty dict on any read / parse failure so callers can
    treat it as optional metadata and still render without it.
    """
    try:
        import gemmi
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block()

        res = None
        for tag in (
            "_refine.ls_d_res_high",
            "_reflns.d_resolution_high",
            "_em_3d_reconstruction.resolution",
        ):
            v = block.find_value(tag)
            if v and v not in ("?", "."):
                try:
                    res = float(v.strip("'\""))
                    break
                except (ValueError, TypeError):
                    pass

        method = None
        v = block.find_value("_exptl.method")
        if v and v not in ("?", "."):
            raw = v.strip("'\"").upper()
            if "X-RAY" in raw:
                method = "X-ray"
            elif "ELECTRON MICROSCOPY" in raw:
                method = "cryo-EM"
            elif "NMR" in raw:
                method = "NMR"
            elif "NEUTRON" in raw:
                method = "neutron"
            else:
                method = raw.title()

        return {"resolution": res, "method": method}
    except Exception:
        return {}


def fetch_alphafold(uniprot_id: str, dest_dir: Path) -> Path:
    """Download AF model into dest_dir, return path. Cached on disk."""
    from ..api.structures import resolve_alphafold_url, _download_file

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{uniprot_id}_AF.cif"
    if out.exists():
        return out
    url = resolve_alphafold_url(uniprot_id, fmt="cif")
    _download_file(url, str(out))
    return out


# Cached experimental PDB cifs follow the `<uid>_<pdbid>_<chain>.cif`
# convention from `api.structures._make_filename`. Using a glob on
# that prefix lets a project that's been around since the AF-only
# era keep using its cached AF model, while a fresh project picks up
# any PDB structure dropped into `structures/` automatically.
def cached_structure_path(uniprot_id: str, structures_dir: Path) -> Optional[Path]:
    """Path to a cached structure cif (PDB if present, else AlphaFold).

    Convenience wrapper for callers that only need the path itself —
    the structure-view widget uses `_cached_structure` directly to get
    the source label too. Returns ``None`` when no cif is cached, so
    callers can short-circuit without having to know the AF/PDB
    filename convention.
    """
    res = _cached_structure(uniprot_id, structures_dir)
    return res[0] if res is not None else None


def _cached_structure(
    uniprot_id: str, structures_dir: Path,
    preferred: Optional[str] = None,
) -> Optional[Tuple[Path, str]]:
    """Return ``(path, source_label)`` of a cached cif, preferring PDB.

    ``preferred`` (basename string) takes precedence when set — this
    is how the structures-gallery's "Make default" action overrides
    the alphabetic PDB pick. Source label is ``"PDB <pdbid>_<chain>"``
    for experimental structures (parsed from the filename), or
    ``"AlphaFold"`` for the AF prediction. Returns ``None`` if no
    cached cif at all.
    """
    if not structures_dir.exists():
        return None
    if preferred:
        candidate = structures_dir / preferred
        if candidate.exists():
            stem = candidate.stem
            if stem == f"{uniprot_id}_AF":
                return candidate, "AlphaFold"
            struct_id = (
                stem[len(uniprot_id) + 1:]
                if stem.startswith(f"{uniprot_id}_") else stem
            )
            return candidate, f"PDB {struct_id}"
    af_name = f"{uniprot_id}_AF.cif"
    pdb_cifs = [
        p for p in sorted(structures_dir.glob(f"{uniprot_id}_*.cif"))
        if p.name != af_name
    ]
    if pdb_cifs:
        # Filename layout: `<uid>_<pdbid>[_<chain>].cif` — strip the
        # uid prefix to recover a human-readable source label.
        stem = pdb_cifs[0].stem
        struct_id = stem[len(uniprot_id) + 1:] if stem.startswith(f"{uniprot_id}_") else stem
        return pdb_cifs[0], f"PDB {struct_id}"
    af_cif = structures_dir / af_name
    if af_cif.exists():
        return af_cif, "AlphaFold"
    return None


def find_or_fetch_structure(
    uniprot_id: str, dest_dir: Path,
) -> Tuple[Path, str]:
    """Return a structure cif for ``uniprot_id``, preferring experimental PDB.

    Resolution order:

    1. Cached cif on disk — prefers any PDB-prefixed file over the
       AlphaFold cache so projects that already have a PDB downloaded
       keep using it without a re-fetch.
    2. PDBe SIFTS query — pick the best PDB structure (lowest-res
       X-ray with the largest UniProt coverage) and download it.
    3. AlphaFold fallback — only if no PDB structure exists for this
       UniProt id at all.

    Returns ``(path, source_label)`` where source_label is either
    ``"PDB <id>"`` (e.g. ``"PDB 2acp_A"``) or ``"AlphaFold"``. The
    label is suitable for the structure-view border title and for
    showing in the chimerax export alongside the colorbar.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cached = _cached_structure(uniprot_id, dest_dir)
    if cached is not None:
        return cached

    # No cache — query PDBe SIFTS for an experimental structure.
    # `_select_structures(selection="best")` picks one per source by
    # method-priority + resolution + coverage, exactly the ranking we
    # want for "give me the best PDB available".
    from ..api.structures import find_structures, fetch_structures

    try:
        df = find_structures([uniprot_id], source="pdb")
    except Exception:
        df = None
    if df is not None and not df.empty:
        try:
            fetched = fetch_structures(
                df, output_dir=str(dest_dir),
                selection="best", skip_existing=True,
            )
            ok = fetched[fetched["local_path"].notna()]
            if not ok.empty:
                row = ok.iloc[0]
                struct_id = row["structure_id"]
                chain = row.get("chain_id") or ""
                label = f"PDB {struct_id}"
                if chain and chain != "-":
                    label = f"PDB {struct_id}_{chain}"
                return Path(row["local_path"]), label
        except Exception:
            # Network hiccup or RCSB rejected the download — fall
            # through to the AF path so the user still sees a model.
            pass

    return fetch_alphafold(uniprot_id, dest_dir), "AlphaFold"


def load_ca_coords(cif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read Cα coords + B-factor (pLDDT for AlphaFold) from first polymer chain.

    Returns (coords [N,3], plddt [N]). Raises ValueError if no polymer.

    For full per-residue parity with the structure (residue numbers,
    one-letter sequence) callers should use :func:`load_ca_residues`,
    which returns the same coords/B-factor arrays plus the parallel
    residue metadata needed to project target-indexed scalars onto
    the cif's residues. This shorter form stays for backward compat
    with existing callers that don't care about the mapping.
    """
    coords, plddt, _, _ = load_ca_residues(cif_path)
    return coords, plddt


def load_ca_residues(
    cif_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read Cα coords, B-factors, residue numbers, and 1-letter sequence.

    Used by the TUI structure view to align target-indexed scalars
    (conservation, SASA, differential) onto the cif's actual residue
    set when a PDB structure has only partial UniProt coverage.

    Returns ``(coords [N, 3], b_iso [N], residue_numbers [N], chain_seq)``
    where ``residue_numbers`` is the per-CA `seqid.num` from the cif
    (1-indexed for SIFTS-aligned chains; arbitrary for older PDBs)
    and ``chain_seq`` is the one-letter ungapped sequence of the
    chain in the same order, suitable for pairwise alignment to the
    target.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    coords = []
    b_iso = []
    res_nums = []
    res_codes = []
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        for res in polymer:
            ca = res.find_atom("CA", '*')
            if ca is not None:
                coords.append((ca.pos.x, ca.pos.y, ca.pos.z))
                b_iso.append(ca.b_iso)
                res_nums.append(int(res.seqid.num))
                res_codes.append(_three_to_one(res.name))
        break
    if not coords:
        raise ValueError(f"No Cα atoms found in {cif_path}")
    return (
        np.asarray(coords, dtype=float),
        np.asarray(b_iso, dtype=float),
        np.asarray(res_nums, dtype=int),
        "".join(res_codes),
    )


# Standard 20-AA three-letter to one-letter map plus a few common
# non-canonical residues (selenomethionine, pyrrolysine) so cif
# sequences with modified residues still produce alignable strings.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine
    "PYL": "O",
    "SEC": "U",
}


def _three_to_one(resname: str) -> str:
    """3-letter residue code → 1-letter, falling back to 'X' for unknowns."""
    return _THREE_TO_ONE.get(resname.upper(), "X")


# Residue names skipped when collecting ligand atoms — water and
# common bulk solvents/ions that aren't biologically interesting
# in a structure-gallery context. Ions like ZN/MG/CA *are* worth
# showing (cofactors), so we don't skip them.
_LIGAND_SKIP = {"HOH", "WAT", "DOD", "TIP", "TIP3", "TIP4"}


def load_ligand_atoms(cif_path: Path):
    """Read non-polymer (HETATM) atom coords for ligand rendering.

    Returns a list of ``{"name": str, "coords": np.ndarray (M, 3)}``
    dicts — one entry per distinct het residue (each ligand instance
    becomes its own group so they can be colored differently). Waters
    are skipped via ``_LIGAND_SKIP``; metal ions are kept (they're
    typically cofactors worth surfacing).

    Empty list when the cif has no non-polymer residues. Used by the
    structure-gallery to overlay bound ligands on the protein render —
    a subtle but useful hint when browsing experimental PDBs vs the
    AlphaFold prediction (which has no ligands).
    """
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    out = []
    for chain in model:
        # `get_ligands()` is gemmi's purpose-built non-polymer iterator;
        # falls back to walking all residues if the binding doesn't
        # expose it on this gemmi version.
        residues = (
            list(chain.get_ligands())
            if hasattr(chain, "get_ligands") else
            [r for r in chain if not r.entity_type == gemmi.EntityType.Polymer]
        )
        for res in residues:
            name = res.name.upper()
            if name in _LIGAND_SKIP:
                continue
            atoms = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
            if not atoms:
                continue
            out.append({
                "name": name,
                "coords": np.asarray(atoms, dtype=float),
            })
    return out


def project_target_to_cif(
    scalar_target: np.ndarray,
    target_seq: str,
    cif_seq: str,
    sentinel: float = 0.0,
) -> np.ndarray:
    """Re-index a target-indexed scalar onto the cif's residue order.

    Used when an experimental PDB structure has only partial UniProt
    coverage (cif has, say, residues 1-98 of a 99-aa target) so the
    cif-indexed coords array and the target-indexed scalar arrays —
    conservation, SASA, differential — would otherwise be a length
    apart and refuse to render.

    Pairwise-aligns ``cif_seq`` to ``target_seq``; for each cif
    residue that aligns to a target position, copies the target
    scalar into the cif slot. Cif residues with no target alignment
    (insertions, modified residues, mismatches outside the target
    coverage) get ``sentinel`` so the renderer paints them neutrally
    instead of crashing on a length mismatch. AF cifs match 1:1
    with the target so the projection is a no-op there.
    """
    n_cif = len(cif_seq)
    if n_cif == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.full(n_cif, sentinel, dtype=np.float32)
    if scalar_target is None or len(scalar_target) == 0:
        return out

    # Fast path: AF case where cif sequence equals target sequence
    # (or is a strict prefix/suffix). One-letter equality on the
    # head is the common case; falls through to pairwise alignment
    # otherwise.
    n_target = min(len(scalar_target), len(target_seq))
    if cif_seq[: n_target] == target_seq[: n_target] and n_cif <= len(scalar_target):
        out[: n_cif] = np.asarray(scalar_target[: n_cif], dtype=np.float32)
        return out

    # Pairwise-align cif_seq to target_seq. BioPython's PairwiseAligner
    # is already a beak dep; the alignment is short (≤1000 residues),
    # cheap. Local mode handles partial-coverage PDB chains naturally.
    try:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1
        best = aligner.align(target_seq, cif_seq)[0]
        target_blocks, cif_blocks = best.aligned
    except Exception:
        # Aligner unavailable / malformed sequence — return sentinel
        # so the render still has *something* and doesn't error out.
        return out

    for (t_start, t_end), (c_start, c_end) in zip(target_blocks, cif_blocks):
        for tp, cp in zip(range(t_start, t_end), range(c_start, c_end)):
            if 0 <= tp < len(scalar_target) and 0 <= cp < n_cif:
                out[cp] = float(scalar_target[tp])
    return out


def parse_secondary_structure(cif_path: Path, seq_length: int) -> str:
    """Return a string of length `seq_length` of {'H','E','C'} per residue.

    Uses the existing mmCIF SS parser from beak.structures.features. AlphaFold
    CIFs include `_struct_conf` (helices) and `_struct_sheet_range` records.
    Falls back to all-coil if parsing fails.
    """
    try:
        from ..structures.features import _parse_chain, _parse_ss_from_mmcif
        _, _, chain_id = _parse_chain(str(cif_path))
        ss_map = _parse_ss_from_mmcif(str(cif_path), chain_id)
    except Exception:
        return 'C' * seq_length
    return ''.join(ss_map.get(i, 'C') for i in range(1, seq_length + 1))


def sasa_color(score: float, max_sasa: float = 100.0) -> str:
    """Buried (low rSASA) → exposed (high rSASA): teal → orange.

    Inputs are percentage **relative SASA** (Tien et al. 2013
    normalisation) in [0, 100]. The default ``max_sasa=100`` reflects
    the [0, 100] %-rSASA convention; the kwarg is retained so callers
    that still pass absolute Å² values won't silently misrender —
    they'll just saturate sooner.
    """
    s = max(0.0, min(max_sasa, float(score)))
    t = s / max_sasa
    # Lerp #2E86AB (teal/buried) → #FF7D45 (orange/exposed)
    r = int(0x2E + (0xFF - 0x2E) * t)
    g = int(0x86 + (0x7D - 0x86) * t)
    b = int(0xAB + (0x45 - 0xAB) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def conservation_color(score: float, midpoint: float = 50.0) -> str:
    """White (low) → red (high) gradient with adjustable midpoint.

    `midpoint` is the score at which the gradient sits at 50% red.
    Shifting it left (e.g. 30) makes more positions red; shifting right
    (e.g. 70) reserves red for only the most conserved positions.
    """
    s = max(0.0, min(100.0, float(score)))
    # Linearly remap so `midpoint` lands at 50, then clamp.
    remapped = max(0.0, min(100.0, s - midpoint + 50.0))
    fade = int(round(255 * (1.0 - remapped / 100.0)))
    return f"#FF{fade:02X}{fade:02X}"


def differential_color(score: float, scale: float = 0.5) -> str:
    """Diverging gradient for signed differential-PSSM scores.

    Input is signed Jensen-Shannon divergence in roughly [-1, 1]. Blue
    means the position is enriched for the *low*-group residue; red
    means enriched for the *high*-group residue; near-zero is the
    neutral panel grey so unaffected positions don't compete visually
    with the differentiated ones. ``scale`` controls how quickly the
    gradient saturates — defaults to 0.5 so values around 0.5 already
    render as fully saturated.
    """
    s = max(-1.0, min(1.0, float(score)))
    t = max(0.0, min(1.0, abs(s) / scale)) if scale > 0 else (1.0 if s else 0.0)
    if s >= 0:
        # neutral grey -> red
        r = int(0x55 + (0xFF - 0x55) * t)
        g = int(0x55 + (0x35 - 0x55) * t)
        b = int(0x55 + (0x35 - 0x55) * t)
    else:
        # neutral grey -> blue
        r = int(0x55 + (0x35 - 0x55) * t)
        g = int(0x55 + (0x82 - 0x55) * t)
        b = int(0x55 + (0xFF - 0x55) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


# Domain palette used by the sequence view's domain bars; we share it
# here so structure-view residues color in lock-step with the sequence
# view when Pfam mode is active.
DOMAIN_PALETTE = ("#FF6B9D", "#FFA62B", "#A66DD4", "#3DCFD4", "#7DD87D")
# Color for residues that fall outside any annotated Pfam domain.
PFAM_NONE_COLOR = "#3A3F4A"


def pfam_color(domain_idx: float) -> str:
    """Map a domain index (or -1 for unannotated) to a palette color."""
    idx = int(domain_idx)
    if idx < 0:
        return PFAM_NONE_COLOR
    return DOMAIN_PALETTE[idx % len(DOMAIN_PALETTE)]


def bfactor_color(score: float, max_b: float = 50.0) -> str:
    """Thermal palette for crystallographic B-factor (Å²): blue → white → red.

    Standard "spectrum b" coloring used by PyMOL (`spectrum b`) and
    ChimeraX (`color byattribute bfactor palette ...`). Low B (rigid
    / well-ordered) is cool blue; mid (~25 Å²) is white; high B
    (mobile / disordered) is hot red. The 0–50 Å² range covers the
    typical band for well-resolved crystal structures; values above
    saturate at red rather than spilling over.

    Distinct from `plddt_color` (which uses AlphaFold's confidence
    bands on a unitless 0–100 scale) — pLDDT and B-factor live in
    the same cif column but mean opposite things, so they get
    opposite gradients.
    """
    s = max(0.0, min(max_b, float(score)))
    t = s / max_b
    if t < 0.5:
        # blue → white
        u = t * 2.0
        r = int(0x33 + (0xFF - 0x33) * u)
        g = int(0x66 + (0xFF - 0x66) * u)
        b = int(0xFF + (0xFF - 0xFF) * u)
    else:
        # white → red
        u = (t - 0.5) * 2.0
        r = int(0xFF + (0xCC - 0xFF) * u)
        g = int(0xFF + (0x00 - 0xFF) * u)
        b = int(0xFF + (0x00 - 0xFF) * u)
    return f"#{r:02X}{g:02X}{b:02X}"


def color_for_mode(score: float, mode: str = "plddt", midpoint: float = 50.0) -> str:
    """Single entry point to map a per-residue score to a color."""
    if mode == "conservation":
        return conservation_color(score, midpoint)
    if mode == "sasa":
        return sasa_color(score)
    if mode == "differential":
        return differential_color(score)
    if mode == "pfam":
        return pfam_color(score)
    if mode == "bfactor":
        return bfactor_color(score)
    return plddt_color(score)


def load_sasa(cif_path: Path, seq_length: int):
    """Per-residue **relative SASA** aligned to the target sequence.

    Returns ``ndarray`` of length ``seq_length`` with values in
    [0, 100] % — observed SASA divided by the per-residue maximum
    from the Tien et al. 2013 table (the field-standard normalization
    reference). The shift to rSASA away from raw Å² is the convention
    used across the literature for "is this residue buried/exposed?"
    questions; absolute SASA varies wildly across residue types
    (Gly max 85 Å² vs Trp max 259 Å²) so a single absolute threshold
    misclassifies small vs large residues. With rSASA, the standard
    classification cutoffs are 20 % (buried) and 50 % (exposed).

    Returns ``None`` on failure (parse error, no chain found).
    """
    try:
        from ..structures.features import _parse_chain, _compute_relative_sasa
        _, _, chain_id = _parse_chain(str(cif_path))
        rsasa_map = _compute_relative_sasa(str(cif_path), chain_id)
    except Exception:
        return None
    return np.asarray(
        [rsasa_map.get(i, 0.0) for i in range(1, seq_length + 1)],
        dtype=np.float32,
    )


def plddt_color(score: float) -> str:
    for threshold, color in _PLDDT_BANDS:
        if score >= threshold:
            return color
    return _PLDDT_BANDS[-1][1]


def _dim_hex(hex_color: str, factor: float) -> str:
    """Scale RGB channels by `factor` (0..1). Used for depth shading."""
    h = hex_color.lstrip("#")
    r = int(int(h[0:2], 16) * factor)
    g = int(int(h[2:4], 16) * factor)
    b = int(int(h[4:6], 16) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


# Default mist tint when the renderer has no explicit background color
# to fade toward (e.g. unit tests, ChimeraX-export-only callers). Same
# hex as `PFAM_NONE_COLOR`, a cool dim gray that reads as "haze" on
# the default Textual dark surface. Real renders pass `bg_color` so
# the fade always pulls toward the actual panel background — the user
# explicitly asked for this so a "white" bg makes far residues fade
# toward white, not gray.
_MIST_RGB_DEFAULT = (0x3A, 0x3F, 0x4A)

# How much mist to mix in at the farthest depth (0 = no mist, 1 = full
# replacement). 0.3 keeps the colour identifiable while clearly
# pushing back-of-structure residues away from the viewer.
_MIST_MAX = 0.30


def _hex_to_rgb(hex_color: str) -> tuple:
    """Parse `#RRGGBB` (or `RRGGBB`) into an `(r, g, b)` int tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _mist_hex(hex_color: str, depth: float, bg_color: str = None) -> str:
    """Lerp an RGB color toward the panel background by `depth * _MIST_MAX`.

    `depth ∈ [0, 1]` where 0 = closest residue (no mist) and 1 =
    farthest residue (max mist). When ``bg_color`` is given (hex
    string), distant residues fade toward that color so the structure
    reads as receding into the panel background rather than into a
    fixed cool-gray tint — works for both dark and light themes. Falls
    back to `_MIST_RGB_DEFAULT` when no bg color is supplied.
    """
    depth = max(0.0, min(1.0, depth))
    t = depth * _MIST_MAX
    fg_r, fg_g, fg_b = _hex_to_rgb(hex_color)
    if bg_color:
        try:
            bg_r, bg_g, bg_b = _hex_to_rgb(bg_color)
        except (ValueError, IndexError):
            bg_r, bg_g, bg_b = _MIST_RGB_DEFAULT
    else:
        bg_r, bg_g, bg_b = _MIST_RGB_DEFAULT
    r = int(fg_r * (1.0 - t) + bg_r * t)
    g = int(fg_g * (1.0 - t) + bg_g * t)
    b = int(fg_b * (1.0 - t) + bg_b * t)
    return f"#{r:02X}{g:02X}{b:02X}"


# Maximum highlight strength — lerp toward white at this fraction at
# peak. Anything stronger blows the residue's hue out to white;
# anything weaker doesn't read as a highlight against the mist.
_HIGHLIGHT_MAX = 0.45


def _highlight_hex(hex_color: str, brightness: float) -> str:
    """Lerp an RGB color toward white by `brightness * _HIGHLIGHT_MAX`.

    Companion to `_mist_hex` but pushes color toward white instead of
    the background — used to add a specular-style top-of-tube highlight
    on the shaded tube view. ``brightness ∈ [0, 1]`` where 0 = no
    highlight (return color as-is) and 1 = strongest. Combined with
    mist, the tube reads as a 3D ribbon with a bright ridge facing
    the viewer and dimmer flanks fading into the background.
    """
    brightness = max(0.0, min(1.0, brightness))
    if brightness <= 0.0:
        return hex_color
    t = brightness * _HIGHLIGHT_MAX
    fg_r, fg_g, fg_b = _hex_to_rgb(hex_color)
    r = int(fg_r * (1.0 - t) + 255 * t)
    g = int(fg_g * (1.0 - t) + 255 * t)
    b = int(fg_b * (1.0 - t) + 255 * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _catmull_rom(
    pts: np.ndarray, steps_per_segment: int = 8
) -> tuple:
    """Catmull-Rom spline through `pts`. Returns (interpolated_points, src_index)
    where src_index[i] gives the nearest original point's index — used for color.
    """
    n = len(pts)
    if n < 2:
        return pts, list(range(n))
    # Pad ends by duplicating first/last for the boundary segments.
    padded = np.vstack([pts[:1], pts, pts[-1:]])

    out: list = []
    src_idx: list = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for s in range(steps_per_segment):
            t = s / steps_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(point)
            src_idx.append(min(n - 1, (i - 1) if t < 0.5 else i))
    out.append(pts[-1])
    src_idx.append(n - 1)
    return np.asarray(out), src_idx


def render_structure(
    coords: np.ndarray,
    plddt: np.ndarray,
    width: int,
    height: int,
    angle_y: float = 0.0,
    angle_x: float = 0.0,
    color_mode: str = "plddt",
    midpoint: float = 50.0,
    view_mode: str = "trace",
    bg_color: str = None,
    ligand_groups: Optional[list] = None,
) -> "Text":
    """Render coords to a braille-art `rich.text.Text` colored by pLDDT.

    Returns a `rich.text.Text` rather than a markup string so the
    Textual compositor doesn't have to re-tokenize ~thousands of
    `[#hex]…[/]` tags on every paint. Pre-styled segments avoid a
    pathological tokenizer regex backtrack that has frozen the UI in
    the past.

    The cell grid is `width × height`; the braille sub-pixel grid is
    `2*width × 4*height`. When multiple residues land on the same dot,
    the highest-z (closer to viewer) wins for color.

    `angle_y` rotates around the vertical axis (spin); `angle_x`
    rotates around the horizontal axis (tilt). Both in radians.
    """
    from rich.text import Text
    if len(coords) == 0 or width < 2 or height < 2:
        return Text("")

    # Center on the protein centroid. Ligand atoms are translated by
    # the same centroid so they stay attached to their parent
    # structure (otherwise a bound ligand would float at the cif's
    # global origin, far from the protein it's on).
    centroid = coords.mean(axis=0)
    pts = coords - centroid

    # Apply rotation to ligand coords as a single concatenated tensor
    # so the rotation block stays unchanged below — split back out
    # before plotting.
    lig_pts_concat = None
    lig_group_sizes: list = []
    if ligand_groups:
        for g in ligand_groups:
            c = g.get("coords")
            if c is None or len(c) == 0:
                continue
            lig_group_sizes.append(len(c))
            lig_pts_concat = (
                c - centroid if lig_pts_concat is None
                else np.vstack([lig_pts_concat, c - centroid])
            )

    # Y-axis rotation (spin)
    if angle_y != 0.0:
        c, s = np.cos(angle_y), np.sin(angle_y)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pts = pts @ ry.T
        if lig_pts_concat is not None:
            lig_pts_concat = lig_pts_concat @ ry.T

    # X-axis rotation (tilt)
    if angle_x != 0.0:
        c, s = np.cos(angle_x), np.sin(angle_x)
        rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pts = pts @ rx.T
        if lig_pts_concat is not None:
            lig_pts_concat = lig_pts_concat @ rx.T

    pix_w = width * 2
    pix_h = height * 4

    # Isotropic fit with 1-pixel margin
    span_x = pts[:, 0].max() - pts[:, 0].min() or 1.0
    span_y = pts[:, 1].max() - pts[:, 1].min() or 1.0
    scale = min((pix_w - 2) / span_x, (pix_h - 2) / span_y)

    # Spline smoothing for "tube" — Catmull-Rom through the Cα atoms turns
    # the polyline into a smooth ribbon-like curve.
    if view_mode == "tube" and len(pts) >= 2:
        sub_pts, sub_idx = _catmull_rom(pts, steps_per_segment=8)
    else:
        sub_pts = pts
        sub_idx = list(range(len(pts)))

    cx, cy = pix_w / 2.0, pix_h / 2.0
    px = (sub_pts[:, 0] * scale + cx).astype(int)
    py = (-sub_pts[:, 1] * scale + cy).astype(int)  # flip y for screen coords
    z = sub_pts[:, 2]

    # z-range for depth shading
    z_min = float(z.min()) if len(z) else 0.0
    z_max = float(z.max()) if len(z) else 1.0
    z_range = max(z_max - z_min, 1e-6)

    # Per-pixel z-buffer + residue-index buffer + highlight buffer.
    # `hbuf` is per-pixel highlight intensity in [0, 1] — only the
    # tube path writes anything > 0 to it (trace stays uniform). The
    # final color step lerps each cell's color toward white by
    # `hbuf[yp, xp] * _HIGHLIGHT_MAX`, producing a bright ridge at the
    # top of the tube that fades to the dim flanks. Combined with the
    # depth-mist this is enough to read as 3D shaded.
    zbuf = np.full((pix_h, pix_w), -np.inf, dtype=float)
    rbuf = np.full((pix_h, pix_w), -1, dtype=int)
    hbuf = np.zeros((pix_h, pix_w), dtype=float)

    def _plot(xp: int, yp: int, zp: float, ridx: int, h: float = 0.0) -> None:
        if 0 <= xp < pix_w and 0 <= yp < pix_h and zp > zbuf[yp, xp]:
            zbuf[yp, xp] = zp
            rbuf[yp, xp] = ridx
            hbuf[yp, xp] = h

    n_pts = len(sub_pts)
    if view_mode == "tube":
        # Dense Catmull-Rom polyline drawn as a 3-track ribbon: a
        # centerline highlight + two flank tracks at ±1 perpendicular.
        # Each sub-segment is linspace-filled so consecutive sub-points
        # never leave pixel gaps — without this, regions where the
        # curve dives through z plot as a noisy point cloud rather
        # than a continuous tube. The center track gets h=1.0 (bright
        # ridge), flanks get h=0.15 with a slightly recessed z so the
        # depth-mist dims them, giving the 3D shaded look.
        for i in range(n_pts - 1):
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
            perp_x, perp_y = -dy / norm, dx / norm
            steps = max(abs(dx), abs(dy)) + 1
            xs = np.linspace(px[i], px[i + 1], steps)
            ys = np.linspace(py[i], py[i + 1], steps)
            zs = np.linspace(z[i], z[i + 1], steps)
            for k in range(steps):
                ridx = sub_idx[i] if k * 2 < steps else sub_idx[i + 1]
                cx_i, cy_i = int(round(xs[k])), int(round(ys[k]))
                _plot(cx_i, cy_i, float(zs[k]), ridx, h=1.0)
                back_z = float(zs[k]) - 0.4
                _plot(
                    int(round(xs[k] + perp_x)),
                    int(round(ys[k] + perp_y)),
                    back_z, ridx, h=0.15,
                )
                _plot(
                    int(round(xs[k] - perp_x)),
                    int(round(ys[k] - perp_y)),
                    back_z, ridx, h=0.15,
                )
    else:
        # Trace mode: line segments between consecutive Cα atoms.
        for i in range(n_pts - 1):
            steps = max(abs(px[i + 1] - px[i]), abs(py[i + 1] - py[i])) + 1
            xs = np.linspace(px[i], px[i + 1], steps).round().astype(int)
            ys = np.linspace(py[i], py[i + 1], steps).round().astype(int)
            zs = np.linspace(z[i], z[i + 1], steps)
            for k in range(steps):
                ridx = sub_idx[i] if k < steps / 2 else sub_idx[i + 1]
                _plot(int(xs[k]), int(ys[k]), float(zs[k]), ridx)

    if n_pts == 1:
        _plot(int(px[0]), int(py[0]), float(z[0]), sub_idx[0])

    # ---- Ligand overlay ----
    # Ligands ride on top of the protein with their own buffer. Each
    # ligand-cell renders bright yellow regardless of color_mode so
    # bound cofactors / substrates are immediately visible against
    # whatever palette the protein is using. Depth-mist applies via
    # the same `lig_z` so a ligand on the far side of the protein
    # still recedes correctly.
    lig_set = np.zeros((pix_h, pix_w), dtype=bool)
    lig_z = np.full((pix_h, pix_w), -np.inf, dtype=float)
    n_ligand_atoms = 0
    if lig_pts_concat is not None and len(lig_pts_concat) > 0:
        lig_px = (lig_pts_concat[:, 0] * scale + cx).astype(int)
        lig_py = (-lig_pts_concat[:, 1] * scale + cy).astype(int)
        lig_zs = lig_pts_concat[:, 2]
        # Update the global z-range so the ligand atoms participate in
        # the same depth-mist as the protein (without this, a far
        # ligand would render at its own local z scale).
        z_min = min(z_min, float(lig_zs.min()))
        z_max = max(z_max, float(lig_zs.max()))
        z_range = max(z_max - z_min, 1e-6)
        for i in range(len(lig_pts_concat)):
            xp, yp = int(lig_px[i]), int(lig_py[i])
            if 0 <= xp < pix_w and 0 <= yp < pix_h:
                # 3×3 sub-pixel splat so each ligand atom reads as a
                # filled circle rather than a single dot — at typical
                # gallery zoom one atom = ~0.5 cell, so a single dot
                # would disappear.
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        xx, yy = xp + ddx, yp + ddy
                        if (
                            0 <= xx < pix_w and 0 <= yy < pix_h
                            and float(lig_zs[i]) > lig_z[yy, xx]
                        ):
                            lig_set[yy, xx] = True
                            lig_z[yy, xx] = float(lig_zs[i])
                n_ligand_atoms += 1

    # Build a Rich `Text` directly rather than a markup string. Returning
    # a markup string from `widget.render()` makes Textual's compositor
    # call `Content.from_markup` on every paint, which runs a regex
    # tokenizer over the (very large, ~thousands of `[#hex]…[/]` tags)
    # output. We hit a pathological case where that regex spun forever
    # and froze the whole UI — see `~/.beak/last-crash.log` watchdog
    # entries. Text segments are pre-styled and skip the parser entirely.
    from rich.style import Style
    from rich.text import Text
    text = Text()
    for cell_row in range(height):
        py0 = cell_row * 4
        if cell_row > 0:
            text.append("\n")
        for cell_col in range(width):
            px0 = cell_col * 2
            mask = 0
            best_z = -np.inf
            best_res = -1
            best_h = 0.0  # peak highlight across the cell's lit dots
            lig_mask = 0  # ligand bits override protein bits
            lig_best_z = -np.inf
            for (dx, dy), bit in _BRAILLE_BITS.items():
                xp = px0 + dx
                yp = py0 + dy
                if lig_set[yp, xp]:
                    lig_mask |= (1 << bit)
                    if lig_z[yp, xp] > lig_best_z:
                        lig_best_z = lig_z[yp, xp]
                if rbuf[yp, xp] >= 0:
                    mask |= (1 << bit)
                    if zbuf[yp, xp] > best_z:
                        best_z = zbuf[yp, xp]
                        best_res = rbuf[yp, xp]
                    if hbuf[yp, xp] > best_h:
                        best_h = float(hbuf[yp, xp])
            # Ligand cells override protein cells: a bound cofactor
            # in front of the protein occludes the backbone behind
            # it. Ligands render in fixed yellow with the same
            # depth-mist as the protein.
            if lig_mask != 0:
                char = chr(0x2800 + (lig_mask | mask))
                depth = 1.0 - (lig_best_z - z_min) / z_range
                color = _mist_hex("#FFD93D", depth, bg_color)
                text.append(char, style=Style(color=color))
                continue
            if mask == 0:
                text.append(" ")
            else:
                char = chr(0x2800 + mask)
                color = color_for_mode(plddt[best_res], color_mode, midpoint)
                # Subtle atmospheric-mist depth fade. Distant residues
                # blend toward the panel background color so the
                # structure recedes into the canvas rather than fading
                # toward a fixed gray. Active in both trace and tube
                # modes; the previous tube-only hard dim toward black
                # was too aggressive — this lerps gently and preserves
                # the residue's hue. Pfam mode gets a smaller dose
                # because its categorical palette doesn't have a
                # gradient story to interrupt.
                depth = 1.0 - (best_z - z_min) / z_range
                if color_mode == "pfam":
                    depth *= 0.5
                color = _mist_hex(color, depth, bg_color)
                # Tube highlight: top-of-tube pixels lerp toward
                # white. `best_h` is 0 for trace (no highlight, no
                # change) and up to 1 for the tube ridge.
                if best_h > 0.0:
                    color = _highlight_hex(color, best_h)
                text.append(char, style=Style(color=color))
    return text
