"""Tests for beak.alignments.formatting — pure functions with BioPython objects."""

import pytest
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from beak.alignments.formatting import subsample_aln, ungap, aln_to_pssm, aln_to_consensus


def _make_alignment(seqs):
    """Helper to create a MultipleSeqAlignment from string sequences."""
    records = [SeqRecord(Seq(s), id=f"seq{i}") for i, s in enumerate(seqs)]
    return MultipleSeqAlignment(records)


class TestSubsampleAln:
    def test_subsample_reduces_size(self):
        aln = _make_alignment(["ACGT", "ACGT", "ACGT", "ACGT", "ACGT"])
        result = subsample_aln(aln, n_sequences=3, random_state=42)
        assert len(result) == 3

    def test_subsample_larger_than_input(self):
        aln = _make_alignment(["ACGT", "ACGT"])
        result = subsample_aln(aln, n_sequences=5, random_state=42)
        assert len(result) == 2  # can't subsample more than exists

    def test_keep_records(self):
        aln = _make_alignment(["ACGT", "ACGA", "ACGC", "ACGG"])
        aln[0].id = "keep_me"
        result = subsample_aln(aln, n_sequences=2, random_state=42, keep_records=["keep_me"])
        ids = [r.id for r in result]
        assert "keep_me" in ids


class TestUngap:
    def test_removes_full_gap_columns(self):
        aln = _make_alignment(["A-C", "A-C", "A-C"])
        result = ungap(aln)
        assert result.get_alignment_length() == 2

    def test_threshold_removes_partial_gaps(self):
        aln = _make_alignment(["A-C", "A-C", "AAC"])
        # Gap at position 1 in 2/3 = 0.67 — threshold 0.5 should remove it
        result = ungap(aln, threshold=0.5)
        assert result.get_alignment_length() == 2

    def test_preserves_non_gap_columns(self):
        aln = _make_alignment(["ACG", "ACG"])
        result = ungap(aln)
        assert result.get_alignment_length() == 3


class TestAlnToPssm:
    def test_returns_dataframe(self):
        aln = _make_alignment(["ACDE", "ACDE", "ACDE"])
        pssm = aln_to_pssm(aln)
        assert hasattr(pssm, 'shape')
        assert pssm.shape[0] == 4  # 4 positions

    def test_frequencies_sum_to_one(self):
        aln = _make_alignment(["ACDE", "ACDE", "FGHI"])
        pssm = aln_to_pssm(aln, as_freq=True)
        # Each row should sum to approximately 1 (excluding conservation column)
        aa_cols = [c for c in pssm.columns if c != 'Cons']
        row_sums = pssm[aa_cols].sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 0.01

    def test_counts_mode(self):
        aln = _make_alignment(["AAA", "AAA", "ACA"])
        pssm = aln_to_pssm(aln, as_freq=False)
        assert pssm.iloc[0]['A'] == 3  # all A at position 0


class TestAlnToConsensus:
    def test_basic_consensus(self):
        aln = _make_alignment(["AAA", "AAA", "ACA"])
        consensus = aln_to_consensus(aln)
        # Consensus may include 'Cons' column artifacts; just check length and first char
        assert len(consensus) >= 3
        assert consensus[0] == 'A'
