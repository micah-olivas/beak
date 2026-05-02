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


def load_ca_coords(cif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read Cα coords + B-factor (pLDDT for AlphaFold) from first polymer chain.

    Returns (coords [N,3], plddt [N]). Raises ValueError if no polymer.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    coords = []
    plddt = []
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        for res in polymer:
            ca = res.find_atom("CA", '*')
            if ca is not None:
                coords.append((ca.pos.x, ca.pos.y, ca.pos.z))
                plddt.append(ca.b_iso)
        break
    if not coords:
        raise ValueError(f"No Cα atoms found in {cif_path}")
    return np.asarray(coords, dtype=float), np.asarray(plddt, dtype=float)


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


def sasa_color(score: float, max_sasa: float = 150.0) -> str:
    """Buried (low SASA) → exposed (high SASA): teal → orange.

    `max_sasa` is the value treated as "fully exposed" (Å²). Typical
    surface-exposed residues are 100–150; large/branched can exceed.
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


def color_for_mode(score: float, mode: str = "plddt", midpoint: float = 50.0) -> str:
    """Single entry point to map a per-residue score to a color."""
    if mode == "conservation":
        return conservation_color(score, midpoint)
    if mode == "sasa":
        return sasa_color(score)
    if mode == "differential":
        return differential_color(score)
    return plddt_color(score)


def load_sasa(cif_path: Path, seq_length: int):
    """Per-residue SASA aligned to the target sequence (length N).

    Returns numpy array of length seq_length, or None on failure.
    """
    try:
        from ..structures.features import _parse_chain, _compute_sasa
        _, _, chain_id = _parse_chain(str(cif_path))
        sasa_map = _compute_sasa(str(cif_path), chain_id)
    except Exception:
        return None
    return np.asarray(
        [sasa_map.get(i, 0.0) for i in range(1, seq_length + 1)],
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
) -> str:
    """Render coords to a braille-art string colored by pLDDT.

    The cell grid is `width × height`; the braille sub-pixel grid is
    `2*width × 4*height`. When multiple residues land on the same dot,
    the highest-z (closer to viewer) wins for color.

    `angle_y` rotates around the vertical axis (spin); `angle_x`
    rotates around the horizontal axis (tilt). Both in radians.
    """
    if len(coords) == 0 or width < 2 or height < 2:
        return ""

    # Center on centroid
    pts = coords - coords.mean(axis=0)

    # Y-axis rotation (spin)
    if angle_y != 0.0:
        c, s = np.cos(angle_y), np.sin(angle_y)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pts = pts @ ry.T

    # X-axis rotation (tilt)
    if angle_x != 0.0:
        c, s = np.cos(angle_x), np.sin(angle_x)
        rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pts = pts @ rx.T

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

    # Per-pixel z-buffer + residue-index buffer
    zbuf = np.full((pix_h, pix_w), -np.inf, dtype=float)
    rbuf = np.full((pix_h, pix_w), -1, dtype=int)

    def _plot(xp: int, yp: int, zp: float, ridx: int) -> None:
        if 0 <= xp < pix_w and 0 <= yp < pix_h and zp > zbuf[yp, xp]:
            zbuf[yp, xp] = zp
            rbuf[yp, xp] = ridx

    n_pts = len(sub_pts)
    if view_mode == "tube":
        # Dense interpolated polyline + ±1 perpendicular offsets so the
        # trace reads as a 2-3 pixel ribbon rather than a wire.
        for i in range(n_pts):
            _plot(int(px[i]), int(py[i]), float(z[i]), sub_idx[i])
            if i + 1 < n_pts:
                dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            elif i > 0:
                dx, dy = px[i] - px[i - 1], py[i] - py[i - 1]
            else:
                dx, dy = 0, 0
            norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
            perp_x, perp_y = -dy / norm, dx / norm
            back_z = float(z[i]) - 0.4
            _plot(int(px[i] + perp_x), int(py[i] + perp_y), back_z, sub_idx[i])
            _plot(int(px[i] - perp_x), int(py[i] - perp_y), back_z, sub_idx[i])
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

    out_lines = []
    for cell_row in range(height):
        parts = []
        py0 = cell_row * 4
        for cell_col in range(width):
            px0 = cell_col * 2
            mask = 0
            best_z = -np.inf
            best_res = -1
            for (dx, dy), bit in _BRAILLE_BITS.items():
                xp = px0 + dx
                yp = py0 + dy
                if rbuf[yp, xp] >= 0:
                    mask |= (1 << bit)
                    if zbuf[yp, xp] > best_z:
                        best_z = zbuf[yp, xp]
                        best_res = rbuf[yp, xp]
            if mask == 0:
                parts.append(" ")
            else:
                char = chr(0x2800 + mask)
                color = color_for_mode(plddt[best_res], color_mode, midpoint)
                if view_mode == "tube":
                    # Depth shading — far residues fade toward black.
                    factor = 0.35 + 0.65 * (best_z - z_min) / z_range
                    color = _dim_hex(color, factor)
                parts.append(f"[{color}]{char}[/{color}]")
        out_lines.append("".join(parts))
    return "\n".join(out_lines)
