"""Per-column conservation scoring for multiple sequence alignments.

Single dispatcher (`conservation_score`) so every consumer in beak —
the TUI structure overlay, the PSSM exporter, the comparative analysis
module — reads from one place and stays consistent when the underlying
metric is changed or tuned.

Default method is **Jensen-Shannon divergence against a BLOSUM62-derived
background distribution**, the field-standard conservation score from
Capra & Singh, *Bioinformatics* 23, 1875–1882 (2007). Validated as the
top performer in multiple comparative studies for identifying
functionally important residues:

  * Johansson & Toh, *BMC Bioinformatics* (2010) — compared 25 methods
    against the Catalytic Site Atlas; JSD won.
  * Wang & Samudrala, *BMC Bioinformatics* (2006) — incorporating a
    background AA frequency improved entropy-based measures.
  * Sinha, *bioRxiv* (2025) — JSD precision 0.758 vs. plain Shannon
    entropy 0.686 vs. SP-BLOSUM62 0.592 on functional-site detection.

Other methods kept reachable for backward compatibility and ablation:

  * ``shannon``           — normalized Shannon entropy. Beak's pre-2026
    conservation score for the PSSM 'Cons' column. Doesn't use a
    background distribution, so common residues (L, A) inflate the
    apparent conservation.
  * ``property_entropy``  — Shannon entropy on Mirny & Shakhnovich
    physicochemical classes (10 classes, AAs grouped by property).
    Captures positions where chemistry is preserved even when the
    exact residue isn't (e.g. all-aliphatic columns).
  * ``target_identity``   — fraction of non-gap homologs that match
    the target at each column. Beak's pre-2026 TUI conservation
    metric. Simple, but ignores the *which* residues are present —
    a column of `LLLLLLT` and a column of `LLLLLT` score similarly,
    even though the latter is the more interesting one functionally.
"""

from typing import Iterable, Optional, Union

import numpy as np


# Standard 20 amino acids in canonical alphabet order. Anything outside
# this set (including `-`, `.`, `X`, `B`, `Z`, etc.) is treated as a
# gap/unknown — counted toward the gap fraction but excluded from the
# AA distribution feeding the score itself.
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {aa: i for i, aa in enumerate(_AA)}


# BLOSUM62 background amino acid frequencies (Henikoff & Henikoff
# 1992) — the reference distribution Capra & Singh used. Other
# backgrounds (UniProt's compositional baseline, an in-alignment
# average per Mihalek 2007) can be passed via the ``background``
# kwarg if needed.
_BLOSUM62_BG = np.array([
    0.0742,  # A
    0.0247,  # C
    0.0536,  # D
    0.0631,  # E
    0.0397,  # F
    0.0742,  # G
    0.0226,  # H
    0.0679,  # I
    0.0596,  # K
    0.0986,  # L
    0.0244,  # M
    0.0446,  # N
    0.0494,  # P
    0.0342,  # Q
    0.0517,  # R
    0.0683,  # S
    0.0541,  # T
    0.0709,  # V
    0.0136,  # W
    0.0301,  # Y
])
_BLOSUM62_BG = _BLOSUM62_BG / _BLOSUM62_BG.sum()


# Mirny & Shakhnovich physicochemical classes — for ``property_entropy``.
# Same grouping as the canonical analysis (PNAS 1999); residues that
# share a class register as identical for entropy purposes.
_PROPERTY_CLASSES = {
    "AVLI": 0,   # aliphatic
    "FYW": 1,    # aromatic
    "ST":  2,    # hydroxyl
    "CM":  3,    # sulfur-containing
    "DE":  4,    # acidic
    "NQ":  5,    # amide
    "KR":  6,    # basic
    "H":   7,    # imidazole / histidine
    "P":   8,    # proline (rigid)
    "G":   9,    # glycine (flexible)
}
_AA_TO_PROPERTY = np.full(len(_AA), -1, dtype=np.int8)
for _cls_str, _cls_idx in _PROPERTY_CLASSES.items():
    for _aa in _cls_str:
        _AA_TO_PROPERTY[_AA_IDX[_aa]] = _cls_idx


_DEFAULT_PSEUDOCOUNT = 1e-7  # avoid log(0) without distorting freqs


def conservation_score(
    alignment,
    *,
    method: str = "js_divergence",
    background: Union[str, np.ndarray, dict] = "blosum62",
    window_size: int = 1,
    gap_penalty: bool = True,
    pseudocount: float = _DEFAULT_PSEUDOCOUNT,
    target_seq: Optional[str] = None,
) -> np.ndarray:
    """Per-column conservation score in [0, 1].

    Args:
        alignment: A ``MultipleSeqAlignment``, a list/tuple of
            sequence strings (all the same length), or a 2D ``ndarray``
            of single-character strings shaped ``(n_seq, n_col)``.
        method: One of ``js_divergence`` (Capra & Singh, default),
            ``shannon``, ``property_entropy``, ``target_identity``.
        background: Reference distribution for ``js_divergence``.
            ``"blosum62"`` uses the Henikoff & Henikoff 1992 frequencies;
            ``"uniform"`` gives a flat 1/20 background; pass an explicit
            dict ``{aa: freq}`` or a 20-element array to override.
        window_size: Window width (in columns) for uniform smoothing.
            ``1`` disables smoothing; Capra & Singh recommend ``3``
            for cleaner per-residue signal. Window centers on each
            column; edges shrink the window rather than padding.
        gap_penalty: If True, scale each column's score by its
            non-gap fraction so heavily-gapped columns produce a muted
            conservation value rather than a misleadingly high one
            from the few residues that are present.
        pseudocount: Tiny constant added to AA counts before
            normalization. Keeps log-domain math finite for columns
            where some AAs are absent; default is small enough to be
            invisible in practice.
        target_seq: Required for ``method='target_identity'``. The
            ungapped target sequence used to identify which residue
            counts as "matching" at each column.

    Returns:
        A 1-D ``ndarray`` of length ``n_columns`` with values in [0, 1]
        (higher = more conserved). Columns that are all-gap return 0.
    """
    aln_array = _to_array(alignment)
    n_seq, n_col = aln_array.shape
    if n_col == 0:
        return np.zeros(0, dtype=np.float32)

    counts = _column_aa_counts(aln_array)        # (n_col, 20)
    aa_total = counts.sum(axis=1)                # non-gap residues per column
    gap_fraction = 1.0 - aa_total / n_seq        # in [0, 1]

    # Frequency table with pseudocount smoothing (avoid log(0)).
    safe_total = np.where(aa_total > 0, aa_total, 1.0)
    freqs = counts / safe_total[:, None]
    if pseudocount > 0:
        freqs = (freqs + pseudocount) / (1.0 + pseudocount * len(_AA))

    if method == "js_divergence":
        bg = _resolve_background(background)
        scores = _jensen_shannon(freqs, bg)
    elif method == "shannon":
        scores = _shannon(freqs)
    elif method == "property_entropy":
        scores = _property_entropy(freqs)
    elif method == "target_identity":
        if target_seq is None:
            raise ValueError(
                "method='target_identity' requires the `target_seq` kwarg"
            )
        scores = _target_identity(aln_array, target_seq)
    else:
        raise ValueError(
            f"Unknown conservation method: {method!r}. Available: "
            f"js_divergence, shannon, property_entropy, target_identity."
        )

    # All-gap columns: score is undefined; force to 0 (unconserved).
    scores = np.where(aa_total > 0, scores, 0.0)

    if gap_penalty:
        scores = scores * (1.0 - gap_fraction)

    if window_size and window_size > 1:
        scores = _window_smooth(scores, window_size)

    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-method scorers
# ---------------------------------------------------------------------------


def _jensen_shannon(freqs: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Capra & Singh 2007 JSD against a fixed background.

    JSD(P || Q) = 0.5·KL(P||M) + 0.5·KL(Q||M), where M = 0.5·(P+Q).
    Using log base 2, JSD is bounded above by 1, so the output is
    already in [0, 1]; we clip on float drift.
    """
    m = 0.5 * (freqs + bg)
    jsd = 0.5 * (_kl_div(freqs, m) + _kl_div(bg, m))
    return np.clip(jsd, 0.0, 1.0)


def _kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(P||Q) per row in bits. Defines 0·log(0/q) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, p / q, 1.0)
        log_ratio = np.where(p > 0, np.log2(ratio), 0.0)
        return (p * log_ratio).sum(axis=-1)


def _shannon(freqs: np.ndarray) -> np.ndarray:
    """1 − H/H_max with H = column Shannon entropy and H_max = log2(20)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(freqs > 0, np.log2(freqs), 0.0)
    entropy = -(freqs * log_p).sum(axis=1)
    return 1.0 - entropy / np.log2(len(_AA))


def _property_entropy(freqs: np.ndarray) -> np.ndarray:
    """1 − H/H_max on the 10 Mirny & Shakhnovich property classes."""
    n_classes = int(_AA_TO_PROPERTY.max()) + 1
    class_freqs = np.zeros((freqs.shape[0], n_classes), dtype=np.float64)
    for aa_i in range(len(_AA)):
        class_freqs[:, int(_AA_TO_PROPERTY[aa_i])] += freqs[:, aa_i]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(class_freqs > 0, np.log2(class_freqs), 0.0)
    entropy = -(class_freqs * log_p).sum(axis=1)
    return 1.0 - entropy / np.log2(n_classes)


def _target_identity(aln_array: np.ndarray, target_seq: str) -> np.ndarray:
    """Fraction of non-gap homologs whose residue matches the target.

    Mirrors beak's pre-2026 TUI conservation: the target's residue at
    each column is the reference, and the score is the fraction of
    other sequences with the same residue. The target row is found
    by matching its ungapped sequence to ``target_seq`` (so an MSA
    that put the target somewhere other than row 0 still works).

    Raises:
        ValueError: when the target sequence is not present in the
            alignment by ungapped equality. Pre-2026 builds silently
            fell back to row 0 in this case, which silently mis-
            attributed scores when the aligner had reordered the MSA.
            Callers that previously tolerated the silent fallback
            must now wrap the call in try/except and decide whether
            to skip, re-align, or surface the inconsistency.
    """
    n_seq, n_col = aln_array.shape
    target_clean = target_seq.upper().replace("-", "").replace(".", "")
    target_row = None
    for i in range(n_seq):
        ungapped = "".join(aln_array[i]).replace("-", "").replace(".", "")
        if ungapped == target_clean:
            target_row = i
            break
    if target_row is None:
        # Hard fail. The earlier "fall back to row 0" silently
        # computed conservation against whichever sequence happened
        # to be first in the MSA — sometimes the right answer (when
        # the target really is row 0), often catastrophically wrong
        # (when an external aligner shuffled the rows). The score
        # numbers look plausible either way, so the bug only
        # surfaces downstream as nonsense biology. Refuse to proceed
        # and let the caller decide whether to skip the score, fix
        # the alignment, or pin the target row explicitly.
        raise ValueError(
            "target sequence not found in alignment by ungapped match — "
            "refusing to silently fall back to row 0. The MSA may have "
            "been re-ordered by the aligner, or the target sequence "
            "passed in differs from the one embedded in the alignment."
        )
    target_aln = aln_array[target_row]

    out = np.zeros(n_col, dtype=np.float64)
    for col in range(n_col):
        ref = target_aln[col]
        if ref in ("-", "."):
            continue
        col_chars = aln_array[:, col]
        non_gap = (col_chars != "-") & (col_chars != ".")
        n_non_gap = int(non_gap.sum())
        if n_non_gap == 0:
            continue
        n_match = int(((col_chars == ref) & non_gap).sum())
        out[col] = n_match / n_non_gap
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_array(alignment) -> np.ndarray:
    """Coerce a MultipleSeqAlignment / list-of-strings / ndarray to (n_seq, n_col)."""
    if isinstance(alignment, np.ndarray):
        return alignment
    if hasattr(alignment, "get_alignment_length"):
        # Bio.Align.MultipleSeqAlignment
        return np.array(
            [list(str(rec.seq).upper()) for rec in alignment],
            dtype="U1",
        )
    # Fall through: list/tuple of strings.
    seqs = list(alignment)
    if not seqs:
        return np.zeros((0, 0), dtype="U1")
    width = len(seqs[0])
    return np.array(
        [list(s.upper().ljust(width, "-")[:width]) for s in seqs],
        dtype="U1",
    )


def _column_aa_counts(aln_array: np.ndarray) -> np.ndarray:
    """Per-column standard-AA counts. Anything outside the 20 is ignored."""
    n_col = aln_array.shape[1]
    counts = np.zeros((n_col, len(_AA)), dtype=np.float64)
    for aa, idx in _AA_IDX.items():
        counts[:, idx] = (aln_array == aa).sum(axis=0)
    return counts


def _resolve_background(bg: Union[str, np.ndarray, dict]) -> np.ndarray:
    if isinstance(bg, str):
        if bg.lower() == "blosum62":
            return _BLOSUM62_BG
        if bg.lower() == "uniform":
            return np.full(len(_AA), 1.0 / len(_AA))
        raise ValueError(f"Unknown background distribution: {bg!r}")
    if isinstance(bg, dict):
        out = np.zeros(len(_AA), dtype=np.float64)
        for aa, p in bg.items():
            i = _AA_IDX.get(aa.upper())
            if i is not None:
                out[i] = float(p)
        s = out.sum()
        return out / s if s > 0 else _BLOSUM62_BG
    arr = np.asarray(bg, dtype=np.float64)
    if arr.shape != (len(_AA),):
        raise ValueError(
            f"Background array must have shape (20,), got {arr.shape}"
        )
    s = arr.sum()
    return arr / s if s > 0 else _BLOSUM62_BG


def _window_smooth(scores: np.ndarray, window: int) -> np.ndarray:
    """Uniform smoothing over `window` columns, edges shrink (no padding)."""
    if window <= 1:
        return scores
    radius = window // 2
    n = scores.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = scores[lo:hi].mean()
    return out


# ---------------------------------------------------------------------------
# Convenience: project per-column scores onto target-sequence positions
# ---------------------------------------------------------------------------


def project_to_target(
    column_scores: np.ndarray,
    target_aln_seq: str,
    target_len: int,
) -> np.ndarray:
    """Map per-column conservation onto target-sequence positions.

    For each non-gap residue in ``target_aln_seq`` (the target's row of
    the MSA), copy the column's score to the corresponding target
    position. Useful when the consumer wants per-target-residue values
    rather than per-alignment-column values (e.g. for coloring the
    target's structure).
    """
    out = np.zeros(target_len, dtype=np.float32)
    target_pos = 0
    for col, target_aa in enumerate(target_aln_seq):
        if target_aa in ("-", "."):
            continue
        if target_pos >= target_len:
            break
        if col < len(column_scores):
            out[target_pos] = column_scores[col]
        target_pos += 1
    return out
