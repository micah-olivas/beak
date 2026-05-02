"""Position-resolved comparison between homolog groups.

The motivating use case is temperature adaptation — split an MSA into
high- vs low-growth-temp groups, ask which residues are differentially
enriched. The same machinery handles any continuous trait (pH, salinity,
genome size) by thresholding, or any categorical trait by passing a
group label per sequence.

Output of `differential_pssm` is a (positions × amino acids) frame of
log2 enrichment ratios — positive values mean the AA is over-represented
in the *high* group. `position_enrichment` collapses that to a single
scalar per alignment column (Jensen-Shannon divergence). For structure
coloring we need one value per *target* residue, so
`target_position_scores` walks the gap pattern of the target row and
projects column-level scores back to native positions.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
# Pseudocount added per AA per group to avoid log(0). 0.5 is the standard
# Laplace smoothing for protein PSSMs and damps small-group noise.
_PSEUDO = 0.5


def _records_to_array(
    records: Sequence[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray]:
    """Vectorise a list of (id, gapped_seq) into a 2D char array."""
    if not records:
        raise ValueError("Empty record list")
    L = len(records[0][1])
    for sid, seq in records:
        if len(seq) != L:
            raise ValueError(
                f"Sequence {sid!r} length {len(seq)} differs from {L}"
            )
    ids = [sid for sid, _ in records]
    arr = np.array([list(seq.upper()) for _, seq in records], dtype="<U1")
    return ids, arr


def _counts_for_group(arr: np.ndarray) -> np.ndarray:
    """Count each amino acid per column. Shape: (L, |alphabet|)."""
    L = arr.shape[1]
    counts = np.zeros((L, len(_ALPHABET)), dtype=np.float64)
    for i, aa in enumerate(_ALPHABET):
        counts[:, i] = (arr == aa).sum(axis=0)
    return counts


def _to_freqs(counts: np.ndarray, pseudo: float = _PSEUDO) -> np.ndarray:
    """Smoothed frequency table — adds pseudocount, normalises per row."""
    smoothed = counts + pseudo
    totals = smoothed.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return smoothed / totals


def group_pssm(
    records: Sequence[Tuple[str, str]],
    groups: Sequence[Optional[str]],
    as_freq: bool = True,
    pseudo: float = _PSEUDO,
) -> Dict[str, pd.DataFrame]:
    """Per-group PSSMs — one DataFrame per category in ``groups``.

    Args:
        records: ``[(seq_id, gapped_sequence), ...]``. All sequences must
            share the same length.
        groups: Group label per record (None to drop the row).
        as_freq: Return (smoothed) frequencies if True, raw counts if False.
        pseudo: Laplace pseudocount applied when ``as_freq=True``.

    Returns:
        ``{group_label: DataFrame[L × |AA|]}``
    """
    ids, arr = _records_to_array(records)
    if len(groups) != len(ids):
        raise ValueError(
            f"groups length {len(groups)} != records {len(ids)}"
        )
    out: Dict[str, pd.DataFrame] = {}
    by_group: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        by_group.setdefault(str(g), []).append(i)
    for label, idxs in by_group.items():
        counts = _counts_for_group(arr[idxs])
        mat = _to_freqs(counts, pseudo) if as_freq else counts
        out[label] = pd.DataFrame(mat, columns=list(_ALPHABET))
    return out


def differential_pssm(
    records: Sequence[Tuple[str, str]],
    values: Sequence[Optional[float]],
    threshold: float,
    pseudo: float = _PSEUDO,
) -> pd.DataFrame:
    """Log2 enrichment per position per AA between high/low groups.

    Sequences with ``values[i] >= threshold`` form the "high" group; the
    rest (excluding NaN) form the "low" group.

    Args:
        records: ``[(seq_id, gapped_sequence), ...]``.
        values: Continuous trait value per record; None / NaN drops the row.
        threshold: Split value. ``high = values >= threshold``.
        pseudo: Laplace pseudocount.

    Returns:
        DataFrame indexed by alignment column, columns = standard 20 AAs,
        values = ``log2(freq_high / freq_low)``. Positive ⇒ enriched in
        the high group. Carries a ``.attrs["n_high"]`` / ``["n_low"]``
        tuple for the caller's stats. Raises ``ValueError`` if either
        group ends up empty.
    """
    ids, arr = _records_to_array(records)
    if len(values) != len(ids):
        raise ValueError(
            f"values length {len(values)} != records {len(ids)}"
        )
    high_idx, low_idx = [], []
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(x):
            continue
        (high_idx if x >= threshold else low_idx).append(i)
    if not high_idx or not low_idx:
        raise ValueError(
            f"Threshold split yielded n_high={len(high_idx)}, "
            f"n_low={len(low_idx)} — need both groups non-empty"
        )
    fh = _to_freqs(_counts_for_group(arr[high_idx]), pseudo)
    fl = _to_freqs(_counts_for_group(arr[low_idx]), pseudo)
    log_ratio = np.log2(fh / fl)
    df = pd.DataFrame(log_ratio, columns=list(_ALPHABET))
    df.attrs["n_high"] = len(high_idx)
    df.attrs["n_low"] = len(low_idx)
    df.attrs["threshold"] = float(threshold)
    return df


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row Jensen-Shannon divergence (base 2). Output ∈ [0, 1]."""
    m = 0.5 * (p + q)
    # log2 with masking so 0 entries contribute 0.
    def _kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((a > 0) & (b > 0), a / b, 1.0)
            term = np.where(a > 0, a * np.log2(ratio), 0.0)
        return term.sum(axis=1)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def position_enrichment(
    records: Sequence[Tuple[str, str]],
    values: Sequence[Optional[float]],
    threshold: float,
    pseudo: float = _PSEUDO,
    signed: bool = True,
) -> pd.Series:
    """Per-position differentiation strength (Jensen-Shannon divergence).

    Returns a Series indexed by alignment column. Values in [0, 1] with
    1 meaning the high/low groups have completely disjoint AA usage.

    When ``signed=True`` (the default), the JSD magnitude is signed by
    the dominant log2-enrichment direction at the position — positive
    means the differentiating AA is enriched in the *high* group.
    """
    ids, arr = _records_to_array(records)
    high_idx, low_idx = [], []
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(x):
            continue
        (high_idx if x >= threshold else low_idx).append(i)
    if not high_idx or not low_idx:
        raise ValueError(
            f"Threshold split yielded n_high={len(high_idx)}, "
            f"n_low={len(low_idx)} — need both groups non-empty"
        )
    fh = _to_freqs(_counts_for_group(arr[high_idx]), pseudo)
    fl = _to_freqs(_counts_for_group(arr[low_idx]), pseudo)
    score = _jsd(fh, fl)
    if signed:
        # Sign by the most-enriched AA's direction at each position.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log2(fh / fl)
        # Pick the AA with largest |log_ratio| weighted by freq, so noisy
        # near-zero AAs don't flip the sign on a sparse alphabet.
        weight = np.maximum(fh, fl) * np.abs(log_ratio)
        winner = np.argmax(weight, axis=1)
        sign = np.sign(log_ratio[np.arange(len(score)), winner])
        # JSD is non-negative; multiply by sign (treat 0 → +1).
        sign[sign == 0] = 1
        score = score * sign
    s = pd.Series(score)
    s.attrs["n_high"] = len(high_idx)
    s.attrs["n_low"] = len(low_idx)
    s.attrs["threshold"] = float(threshold)
    return s


def _target_column_map(target_seq: str) -> List[Optional[int]]:
    """For each alignment column, the 0-based target position (or None at gaps)."""
    out: List[Optional[int]] = []
    pos = 0
    for ch in target_seq:
        if ch == "-":
            out.append(None)
        else:
            out.append(pos)
            pos += 1
    return out


def target_position_scores(
    records: Sequence[Tuple[str, str]],
    target_id: str,
    values: Sequence[Optional[float]],
    threshold: float,
    pseudo: float = _PSEUDO,
    signed: bool = True,
) -> np.ndarray:
    """Project per-column enrichment back onto target residue positions.

    Args:
        records: ``[(seq_id, gapped_sequence), ...]``. Must include the
            target as one of the rows (matched by ``target_id``).
        target_id: Identifier of the target sequence among ``records``.
        values: Trait value per record; the target's own value is
            ignored if present.
        threshold, pseudo, signed: Forwarded to ``position_enrichment``.

    Returns:
        np.ndarray of length equal to the target's ungapped sequence,
        with one signed JSD score per residue. Shape matches what the
        structure-view's color machinery already consumes for pLDDT /
        conservation.
    """
    target_seq = None
    for sid, seq in records:
        if sid == target_id:
            target_seq = seq
            break
    if target_seq is None:
        raise ValueError(f"target_id {target_id!r} not found among records")

    col_score = position_enrichment(
        records, values, threshold, pseudo=pseudo, signed=signed
    )
    col_to_pos = _target_column_map(target_seq)
    n_residues = sum(1 for ch in target_seq if ch != "-")
    out = np.zeros(n_residues, dtype=np.float64)
    for col, pos in enumerate(col_to_pos):
        if pos is None:
            continue
        out[pos] = float(col_score.iloc[col])
    return out


def split_indices_by_threshold(
    values: Iterable[Optional[float]],
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """Helper: indices that go into the high vs low groups."""
    high, low = [], []
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(x):
            continue
        (high if x >= threshold else low).append(i)
    return high, low
