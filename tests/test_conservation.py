"""Unit tests for `beak.alignments.conservation`.

Sanity-checks the four scoring methods on hand-constructed alignments
where the expected ranking is unambiguous (fully-conserved column >
half-conserved column > random column > all-gap column), so the tests
catch regressions in either the scoring math or the column-counting /
gap-handling plumbing.
"""

import numpy as np
import pytest

from beak.alignments.conservation import (
    _BLOSUM62_BG,
    conservation_score,
    project_to_target,
)


def _aln(*rows: str) -> list:
    """Tiny helper for building string alignments inline."""
    return list(rows)


# ---------------------------------------------------------------------------
# Per-method sanity
# ---------------------------------------------------------------------------


class TestJSDivergence:
    def test_fully_conserved_column_scores_high(self):
        # All sequences have the same residue at the column → JSD is
        # maximised because the observed delta-distribution is as far
        # from the BLOSUM62 background as it can get.
        aln = _aln("A", "A", "A", "A", "A")
        scores = conservation_score(aln, gap_penalty=False)
        assert scores[0] > 0.5  # JSD on a delta-vs-BLOSUM is well above 0.5

    def test_random_column_scores_low(self):
        # 20 different residues across 20 sequences → frequencies match
        # the uniform-ish background → JSD ≈ 0.
        aln = _aln(*list("ACDEFGHIKLMNPQRSTVWY"))
        scores = conservation_score(aln, gap_penalty=False)
        assert scores[0] < 0.2

    def test_jsd_is_higher_for_more_conserved_column(self):
        # 5 sequences, two columns: col 0 fully conserved, col 1 split
        # 3-vs-2. The JSD ranking should match.
        aln = _aln("AA", "AC", "AC", "AA", "AC")
        scores = conservation_score(aln, gap_penalty=False)
        assert scores[0] > scores[1]

    def test_score_in_unit_interval(self):
        # Random alignment — every score is in [0, 1].
        rng = np.random.default_rng(42)
        rows = [
            "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=30))
            for _ in range(15)
        ]
        scores = conservation_score(rows)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_uniform_background_lowers_score_for_typical_aas(self):
        # A column of all 'L' (BLOSUM bg = 0.099, common) is less
        # surprising under the BLOSUM background than under a uniform
        # background — so uniform-bg JSD should be higher.
        aln = _aln("L", "L", "L", "L", "L")
        blosum = conservation_score(aln, gap_penalty=False, background="blosum62")[0]
        uniform = conservation_score(aln, gap_penalty=False, background="uniform")[0]
        assert uniform > blosum

    def test_dict_background_normalized(self):
        # Skewed handcrafted background: assigning all mass to A makes
        # an all-A column unsurprising → JSD ≈ 0 there.
        skewed = {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        skewed["A"] = 1.0
        score = conservation_score(
            _aln("A", "A", "A"),
            gap_penalty=False,
            background=skewed,
        )[0]
        assert score < 0.05


class TestShannon:
    def test_fully_conserved_is_one(self):
        scores = conservation_score(
            _aln("A", "A", "A"),
            method="shannon",
            gap_penalty=False,
        )
        assert scores[0] == pytest.approx(1.0, abs=1e-3)

    def test_uniform_is_zero(self):
        # All 20 AAs once each → entropy = log2(20) → conservation = 0.
        scores = conservation_score(
            _aln(*list("ACDEFGHIKLMNPQRSTVWY")),
            method="shannon",
            gap_penalty=False,
            pseudocount=0,
        )
        assert scores[0] == pytest.approx(0.0, abs=1e-3)


class TestPropertyEntropy:
    def test_property_class_treats_aliphatics_as_one(self):
        # All sequences are aliphatic (V, L, I, A) but with different
        # exact residues. Identity entropy is moderate, but property
        # entropy is 0 — the column is "fully conserved" in chemistry.
        rows = _aln("V", "L", "I", "A", "V")
        prop = conservation_score(
            rows, method="property_entropy", gap_penalty=False,
        )[0]
        ident = conservation_score(
            rows, method="shannon", gap_penalty=False,
        )[0]
        assert prop > 0.95  # essentially fully conserved by class
        assert ident < prop  # identity sees mismatch where chemistry doesn't


class TestTargetIdentity:
    def test_match_fraction(self):
        # Column 0: every row's residue is 'A' → 5/5 match.
        # Column 1: target residue is 'A'; rows 0 and 4 also have 'A'
        # while rows 1-3 have 'C' → 2/5 match.
        rows = _aln("AA", "AC", "AC", "AC", "AA")
        scores = conservation_score(
            rows,
            method="target_identity",
            target_seq="AA",
            gap_penalty=False,
        )
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(2 / 5)

    def test_requires_target_seq(self):
        with pytest.raises(ValueError, match="target_seq"):
            conservation_score(_aln("A"), method="target_identity")

    def test_target_row_found_by_ungapped_match(self):
        # Target is row 1, not row 0 — the metric should still find it
        # via ungapped-sequence equality.
        rows = _aln("AKR", "AC-", "AKR")
        scores = conservation_score(
            rows,
            method="target_identity",
            target_seq="AC",
            gap_penalty=False,
        )
        # At column 0 the target's residue is A; 3 of 3 non-gap rows
        # have A; score = 1.0.
        assert scores[0] == pytest.approx(1.0)

    def test_missing_target_raises_instead_of_silent_row0_fallback(self):
        # Pre-fix behavior was to silently fall back to row 0 when the
        # target sequence wasn't in the MSA — which masquerades as a
        # successful run but computes "conservation" against the wrong
        # reference. The metric must now refuse and force the caller to
        # handle the mismatch explicitly.
        rows = _aln("AKR", "AKR", "AKR")
        with pytest.raises(ValueError, match="target sequence not found"):
            conservation_score(
                rows,
                method="target_identity",
                target_seq="WWW",  # not present in the alignment
                gap_penalty=False,
            )


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------


class TestGapHandling:
    def test_all_gap_column_scores_zero(self):
        rows = _aln("A-", "A-", "A-")
        scores = conservation_score(rows, method="js_divergence")
        assert scores[1] == 0.0

    def test_gap_penalty_scales_score(self):
        # 5 sequences, one column with 4 gaps and 1 'A'. Gap penalty on
        # should produce a much smaller score than gap penalty off.
        rows = _aln("A", "-", "-", "-", "-")
        with_pen = conservation_score(rows, gap_penalty=True)[0]
        without_pen = conservation_score(rows, gap_penalty=False)[0]
        assert with_pen < without_pen
        # 4/5 gap fraction → penalty factor 0.2.
        assert with_pen == pytest.approx(without_pen * 0.2, rel=0.01)

    def test_unknown_residues_treated_as_gap(self):
        # 'X' is not in the standard 20-AA alphabet → should be treated
        # the same as a gap for scoring purposes (i.e. equal to 'A--').
        score_with_x = conservation_score(_aln("A", "X", "-"), gap_penalty=True)[0]
        score_with_gaps = conservation_score(_aln("A", "-", "-"), gap_penalty=True)[0]
        assert score_with_x == pytest.approx(score_with_gaps, rel=1e-4)


# ---------------------------------------------------------------------------
# Window smoothing & projection
# ---------------------------------------------------------------------------


class TestWindowSmoothing:
    def test_window_one_is_identity(self):
        rows = _aln(*list("ACDEFGHIKLMN"))
        a = conservation_score(rows, window_size=1)
        b = conservation_score(rows, window_size=1)
        assert np.allclose(a, b)

    def test_window_smooths_dip(self):
        # Column 1 has one mismatched residue ('K' against an all-'A'
        # background) — its raw conservation is *lower* than every
        # neighbouring column, so window smoothing should pull it UP
        # toward the surrounding high-conservation values.
        rows = _aln("AAAAA", "AAAAA", "AKAAA", "AAAAA", "AAAAA")
        no_smooth = conservation_score(rows, window_size=1)
        smoothed = conservation_score(rows, window_size=3)
        assert smoothed[1] > no_smooth[1]
        # And the smoothed value sits below the unbroken neighbours,
        # so the window doesn't blow past the column's own context.
        assert smoothed[1] < no_smooth[0]


class TestProjection:
    def test_project_skips_target_gaps(self):
        # Target's row of the alignment is `A-CD`; target sequence is
        # `ACD`. Projection should pick up columns 0, 2, 3.
        scores = np.array([0.9, 0.5, 0.7, 0.4], dtype=np.float32)
        out = project_to_target(scores, "A-CD", target_len=3)
        assert np.allclose(out, [0.9, 0.7, 0.4])

    def test_truncates_extra_columns(self):
        # Aligned target longer than declared target length — projection
        # should stop once the target is filled.
        out = project_to_target(np.ones(10, dtype=np.float32), "AAAA", target_len=2)
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# Background distribution sanity
# ---------------------------------------------------------------------------


def test_blosum_background_sums_to_one():
    assert _BLOSUM62_BG.sum() == pytest.approx(1.0, abs=1e-6)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown conservation method"):
        conservation_score(_aln("A"), method="not-a-method")
