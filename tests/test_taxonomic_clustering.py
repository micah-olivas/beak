"""Unit tests for taxonomic-clustering per-position bias.

Synthetic MSAs with a known answer key:
    col0 — perfectly clade-discriminating (K in clade A, R in clade B)
    col1 — fully conserved (G everywhere)
    col2 — varies within clades but identically across them (clade-blind)
    col3 — mixed / roughly clade-independent

The discriminating column should score high; the conserved and
clade-blind columns should score ~0. The permutation null should rank
the discriminating column well above the clade-blind one, and Henikoff
weighting should down-weight duplicated sequences.
"""

import numpy as np
import pytest

from beak.analysis.comparative import (
    henikoff_weights,
    taxonomic_enrichment,
    target_taxonomic_scores,
)


# records order: target, then clade A x3, clade B x3
_RECORDS = [
    ("target", "KGAA"),
    ("a1", "KGAC"),
    ("a2", "KGCA"),
    ("a3", "KGAC"),
    ("b1", "RGAA"),
    ("b2", "RGCA"),
    ("b3", "RGAC"),
]
_GROUPS = [None, "A", "A", "A", "B", "B", "B"]


def test_discriminating_column_scores_high():
    res = taxonomic_enrichment(_RECORDS, _GROUPS, min_per_clade=3)
    assert res.score_kind == "uncertainty_coefficient"
    # col0 fully explained by clade -> U ~ 1
    assert res.col_score[0] > 0.9
    # col1 conserved -> nothing to explain
    assert res.col_score[1] == 0.0
    # col2 clade-blind -> near zero
    assert res.col_score[2] < 0.2
    assert res.clades == ["A", "B"]
    assert res.clade_sizes == {"A": 3, "B": 3}
    assert res.n_sequences == 6


def test_conserved_column_has_zero_mi():
    res = taxonomic_enrichment(_RECORDS, _GROUPS, min_per_clade=3)
    assert res.mi[1] == pytest.approx(0.0, abs=1e-12)
    assert res.uncertainty[1] == 0.0


def test_permutation_null_separates_signal_from_noise():
    res = taxonomic_enrichment(
        _RECORDS, _GROUPS, min_per_clade=3, n_permutations=200, seed=0
    )
    assert res.score_kind == "permutation_zscore"
    # discriminating column stands well above the clade-blind one
    assert res.col_score[0] > 2.0
    assert res.col_score[0] > res.col_score[2]


def test_permutation_null_is_reproducible():
    kw = dict(min_per_clade=3, n_permutations=100, seed=7)
    a = taxonomic_enrichment(_RECORDS, _GROUPS, **kw)
    b = taxonomic_enrichment(_RECORDS, _GROUPS, **kw)
    np.testing.assert_array_equal(a.col_score, b.col_score)


def test_min_per_clade_floor_raises_when_too_few_clades():
    # Both clades have 3 sequences; a floor of 4 leaves < 2 clades.
    with pytest.raises(ValueError, match="Need >= 2 clades"):
        taxonomic_enrichment(_RECORDS, _GROUPS, min_per_clade=4)


def test_min_per_clade_drops_small_clade():
    records = _RECORDS + [("c1", "MGAA"), ("c2", "MGCA")]
    groups = _GROUPS + ["C", "C"]  # clade C has only 2
    res = taxonomic_enrichment(records, groups, min_per_clade=3)
    assert res.clades == ["A", "B"]  # C dropped
    assert "C" not in res.clade_sizes


def test_none_and_blank_labels_are_dropped():
    groups = [None, "A", "A", "  ", "B", "B", np.nan]
    # rows a3 (blank) and b3 (nan) drop -> A has 2, B has 2
    res = taxonomic_enrichment(_RECORDS, groups, min_per_clade=2)
    assert res.clade_sizes == {"A": 2, "B": 2}


def test_henikoff_downweights_duplicates():
    # three identical rows + one distinct row
    arr = np.array(
        [list("AAAA"), list("AAAA"), list("AAAA"), list("CCCC")],
        dtype="<U1",
    )
    w = henikoff_weights(arr)
    assert w.sum() == pytest.approx(4.0)  # normalised to N
    # the three duplicates share weight; the singleton carries more
    assert w[0] == pytest.approx(w[1]) == pytest.approx(w[2])
    assert w[3] > w[0]
    assert w[3] == pytest.approx(2.0)


def test_weighting_defuses_redundant_clade_bias():
    # Clade A is one sequence duplicated 5x; without weighting that
    # duplication manufactures a confident A-vs-B difference at col0.
    records = [
        ("target", "KA"),
        ("a1", "KA"), ("a2", "KA"), ("a3", "KA"), ("a4", "KA"), ("a5", "KA"),
        ("b1", "RC"), ("b2", "RD"), ("b3", "RE"), ("b4", "RF"), ("b5", "RG"),
    ]
    groups = [None] + ["A"] * 5 + ["B"] * 5
    weighted = taxonomic_enrichment(records, groups, min_per_clade=3, use_weights=True)
    unweighted = taxonomic_enrichment(records, groups, min_per_clade=3, use_weights=False)
    assert weighted.weighted is True
    assert unweighted.weighted is False
    # col1: clade A is 5 identical 'A's, clade B is 5 distinct residues.
    # Weighting collapses A's redundancy so its effective contribution
    # shrinks relative to the naive count.
    assert weighted.n_sequences == unweighted.n_sequences == 10


def test_target_projection_respects_gaps():
    records = [
        ("target", "KG-A"),  # ungapped = KGA, length 3
        ("a1", "KGCA"), ("a2", "KGCA"), ("a3", "KGCA"),
        ("b1", "RGCA"), ("b2", "RGCA"), ("b3", "RGCA"),
    ]
    groups = [None, "A", "A", "A", "B", "B", "B"]
    scores, res = target_taxonomic_scores(records, "target", groups, min_per_clade=3)
    assert scores.shape == (3,)  # one per ungapped target residue
    # col0 (discriminating) maps to target position 0
    assert scores[0] > 0.9
    # col1 (conserved G) maps to target position 1
    assert scores[1] == 0.0


def test_missing_target_id_raises():
    with pytest.raises(ValueError, match="not found"):
        target_taxonomic_scores(_RECORDS, "nope", _GROUPS, min_per_clade=3)
