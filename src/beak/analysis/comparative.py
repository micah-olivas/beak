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


# --------------------------------------------------------------------------- #
# Taxonomic clustering — N-group per-position bias
#
# The differential machinery above answers a two-group question ("high vs
# low trait"). Taxonomic clustering is the multi-group generalization:
# partition the MSA into clades (by lineage rank, or any categorical label)
# and ask, per column, *how much of the residue variation is explained by
# clade membership*. That's the mutual information between the clade label
# and the amino acid at the column — the Evolutionary-Trace / SDPpred
# signal: positions conserved within clades but divergent between them.
#
# Two hazards, both handled here:
#   1. Phylogenetic non-independence. 200 near-identical sequences from one
#      genus should not out-vote 3 from another. `henikoff_weights` down-
#      weights redundancy (Henikoff & Henikoff 1994, position-based).
#   2. Finite-sample bias. The plug-in MI estimate is positive even for
#      random labels, and more so for deep columns. A label-shuffle
#      permutation null (`n_permutations > 0`) turns each column's MI into a
#      z-score against what its own entropy + clade sizes would produce by
#      chance.
#
# MI is computed over the 20 standard AAs only (gaps excluded, consistent
# with the PSSM code above) and, unlike the signed two-group JSD, is
# unsigned — with >2 clades there is no single enrichment direction.
# --------------------------------------------------------------------------- #


def henikoff_weights(arr: np.ndarray) -> np.ndarray:
    """Position-based sequence weights (Henikoff & Henikoff 1994).

    For each column, a residue shared by many sequences contributes little
    to each; a rare residue contributes a lot. A sequence's weight is the
    sum of its per-column contributions ``1 / (k_col * n_res)`` where
    ``k_col`` is the number of distinct symbols in the column and
    ``n_res`` the number of sequences carrying this sequence's symbol
    there. Gaps count as a symbol. Weights are normalised to sum to
    ``N`` so a uniform (non-redundant) alignment recovers unit weights.

    Args:
        arr: ``(N, L)`` character array (rows = sequences).

    Returns:
        ``(N,)`` float weights. All-ones if the alignment has no columns.
    """
    n, L = arr.shape
    if n == 0 or L == 0:
        return np.ones(max(n, 1), dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
    for col in range(L):
        column = arr[:, col]
        _, inverse, counts = np.unique(
            column, return_inverse=True, return_counts=True
        )
        k = len(counts)
        # inverse[i] indexes the symbol of sequence i in this column.
        w += 1.0 / (k * counts[inverse])
    total = w.sum()
    if total <= 0:
        return np.ones(n, dtype=np.float64)
    return w / total * n


def _clade_indexed(
    arr: np.ndarray,
    labels: Sequence[str],
    min_per_clade: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Filter rows to well-populated clades and integer-index them.

    Returns ``(sub_arr, clade_idx, ordered_clades)`` where ``sub_arr`` is
    the retained rows, ``clade_idx[i]`` is the 0-based clade of retained
    row ``i``, and ``ordered_clades`` is the sorted clade-label list.
    Raises ``ValueError`` if fewer than two clades survive the floor.
    """
    from collections import Counter

    sizes = Counter(labels)
    good = sorted(c for c, n in sizes.items() if n >= min_per_clade)
    if len(good) < 2:
        raise ValueError(
            f"Need >= 2 clades with >= {min_per_clade} sequences; "
            f"got {len(good)} (sizes: {dict(sizes)})"
        )
    good_set = set(good)
    keep = [i for i, c in enumerate(labels) if c in good_set]
    order = {c: j for j, c in enumerate(good)}
    cidx = np.array([order[labels[i]] for i in keep], dtype=np.int64)
    return arr[keep], cidx, good


def _column_mi(
    aa_stack: np.ndarray,
    cidx: np.ndarray,
    n_clades: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mutual information ``I(clade; AA)`` and AA entropy.

    Args:
        aa_stack: ``(|AA|, m, L)`` weighted indicator — ``aa_stack[a, i, c]``
            is row ``i``'s weight if it has amino acid ``a`` at column
            ``c``, else 0. Weighting is folded in here so permutations
            (which only reshuffle ``cidx``) reuse the same stack.
        cidx: ``(m,)`` clade index per row.
        n_clades: number of distinct clades.

    Returns:
        ``(mi, h_aa)`` each shape ``(L,)``, in bits. ``h_aa`` is invariant
        to clade permutation (it only depends on the AA marginal), so
        callers computing a null need it once.
    """
    m = cidx.shape[0]
    onehot = np.zeros((m, n_clades), dtype=np.float64)
    onehot[np.arange(m), cidx] = 1.0
    # J[l, c, a] = weighted count of clade c / amino acid a at column l.
    J = np.einsum("mc,aml->lca", onehot, aa_stack)
    tot = J.sum(axis=(1, 2), keepdims=True)
    tot = np.where(tot == 0, 1.0, tot)
    Pj = J / tot
    Pc = Pj.sum(axis=2, keepdims=True)          # (L, C, 1)
    Pa = Pj.sum(axis=1, keepdims=True)          # (L, 1, A)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((Pj > 0) & (Pc > 0) & (Pa > 0), Pj / (Pc * Pa), 1.0)
        mi = np.where(Pj > 0, Pj * np.log2(ratio), 0.0).sum(axis=(1, 2))
        h_aa = -np.where(Pa > 0, Pa * np.log2(Pa), 0.0).sum(axis=(1, 2))
    return mi, h_aa


class TaxonomicResult:
    """Per-column output of :func:`taxonomic_enrichment`.

    Attributes:
        col_score: ``(L,)`` score used for coloring — the permutation
            z-score when ``n_permutations > 0``, else the uncertainty
            coefficient ``U(AA|clade) in [0, 1]``.
        score_kind: ``"permutation_zscore"`` or ``"uncertainty_coefficient"``.
        mi: ``(L,)`` raw mutual information (bits).
        uncertainty: ``(L,)`` ``MI / H(AA)`` in ``[0, 1]``; 0 where the
            column is invariant (nothing to explain).
        clades: ordered clade labels retained after the size floor.
        clade_sizes: ``{clade: n_sequences}`` for retained clades.
        n_sequences: total sequences used (sum of retained clade sizes).
        weighted: whether Henikoff weighting was applied.
    """

    def __init__(
        self,
        col_score: np.ndarray,
        score_kind: str,
        mi: np.ndarray,
        uncertainty: np.ndarray,
        clades: List[str],
        clade_sizes: Dict[str, int],
        n_sequences: int,
        weighted: bool,
    ) -> None:
        self.col_score = col_score
        self.score_kind = score_kind
        self.mi = mi
        self.uncertainty = uncertainty
        self.clades = clades
        self.clade_sizes = clade_sizes
        self.n_sequences = n_sequences
        self.weighted = weighted


def taxonomic_enrichment(
    records: Sequence[Tuple[str, str]],
    groups: Sequence[Optional[str]],
    *,
    use_weights: bool = True,
    min_per_clade: int = 3,
    n_permutations: int = 0,
    seed: int = 0,
) -> TaxonomicResult:
    """Per-column residue bias explained by clade membership.

    Args:
        records: ``[(seq_id, gapped_sequence), ...]``, equal length.
        groups: clade label per record; ``None`` / NaN / empty drops the row
            (e.g. the target, or sequences with no taxonomy at this rank).
        use_weights: apply Henikoff sequence weighting (rigor control).
        min_per_clade: clades with fewer sequences are dropped before
            scoring, so between-clade signal isn't read off tiny groups.
        n_permutations: label-shuffle null size. ``0`` returns the
            uncertainty coefficient; ``> 0`` returns a per-column z-score
            of observed MI against the null (rigor control).
        seed: RNG seed for the permutation null (kept fixed so cached
            results and tests are reproducible).

    Returns:
        :class:`TaxonomicResult`.

    Raises:
        ValueError: if fewer than two clades clear ``min_per_clade``.
    """
    ids, arr = _records_to_array(records)
    if len(groups) != len(ids):
        raise ValueError(
            f"groups length {len(groups)} != records {len(ids)}"
        )
    labels: List[Optional[str]] = []
    for g in groups:
        if g is None or (isinstance(g, float) and np.isnan(g)):
            labels.append(None)
        else:
            s = str(g).strip()
            labels.append(s or None)
    # Keep only labelled rows, then index the well-populated clades.
    kept_rows = [i for i, c in enumerate(labels) if c is not None]
    kept_labels = [labels[i] for i in kept_rows]
    sub, cidx, clades = _clade_indexed(
        arr[kept_rows], kept_labels, min_per_clade
    )
    n_clades = len(clades)

    weights = (
        henikoff_weights(sub) if use_weights
        else np.ones(sub.shape[0], dtype=np.float64)
    )
    L = sub.shape[1]
    aa_stack = np.empty((len(_ALPHABET), sub.shape[0], L), dtype=np.float64)
    for a_i, aa in enumerate(_ALPHABET):
        aa_stack[a_i] = (sub == aa) * weights[:, None]

    mi, h_aa = _column_mi(aa_stack, cidx, n_clades)
    with np.errstate(divide="ignore", invalid="ignore"):
        uncertainty = np.where(h_aa > 0, mi / h_aa, 0.0)
    uncertainty = np.clip(uncertainty, 0.0, 1.0)

    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null = np.empty((n_permutations, L), dtype=np.float64)
        perm = cidx.copy()
        for k in range(n_permutations):
            rng.shuffle(perm)
            null[k], _ = _column_mi(aa_stack, perm, n_clades)
        mu = null.mean(axis=0)
        sd = null.std(axis=0)
        col_score = np.where(sd > 0, (mi - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        score_kind = "permutation_zscore"
    else:
        col_score = uncertainty
        score_kind = "uncertainty_coefficient"

    from collections import Counter

    sizes = Counter(kept_labels)
    clade_sizes = {c: int(sizes[c]) for c in clades}
    return TaxonomicResult(
        col_score=col_score,
        score_kind=score_kind,
        mi=mi,
        uncertainty=uncertainty,
        clades=clades,
        clade_sizes=clade_sizes,
        n_sequences=int(sum(clade_sizes.values())),
        weighted=use_weights,
    )


def target_taxonomic_scores(
    records: Sequence[Tuple[str, str]],
    target_id: str,
    groups: Sequence[Optional[str]],
    *,
    use_weights: bool = True,
    min_per_clade: int = 3,
    n_permutations: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, TaxonomicResult]:
    """Project per-column taxonomic scores onto target residue positions.

    Mirrors :func:`target_position_scores`: walks the target row's gap
    pattern so the returned array has one value per *ungapped* target
    residue, matching what the structure-view color machinery consumes.
    The target's own group label is ignored (set it to ``None`` in
    ``groups`` upstream if the target also appears as a homolog).

    Returns ``(per_residue_scores, result)`` so callers get both the
    ribbon array and the clade metadata for display.
    """
    target_seq = None
    for sid, seq in records:
        if sid == target_id:
            target_seq = seq
            break
    if target_seq is None:
        raise ValueError(f"target_id {target_id!r} not found among records")

    result = taxonomic_enrichment(
        records,
        groups,
        use_weights=use_weights,
        min_per_clade=min_per_clade,
        n_permutations=n_permutations,
        seed=seed,
    )
    col_to_pos = _target_column_map(target_seq)
    n_residues = sum(1 for ch in target_seq if ch != "-")
    out = np.zeros(n_residues, dtype=np.float64)
    for col, pos in enumerate(col_to_pos):
        if pos is None:
            continue
        out[pos] = float(result.col_score[col])
    return out, result
