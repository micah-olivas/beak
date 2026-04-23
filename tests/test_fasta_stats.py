"""Tests for FASTA validation + embedding-size estimation helpers."""

import pytest
import click

from beak.cli.submit import (
    _fasta_stats,
    _estimate_embedding_bytes,
    _humanize_bytes,
)


def _write_fasta(tmp_path, content: str):
    p = tmp_path / "test.fasta"
    p.write_text(content)
    return p


class TestFastaStats:
    def test_single_record(self, tmp_path):
        p = _write_fasta(tmp_path, ">seq1\nACDEFGHIKL\n")
        n, total, mx, mn = _fasta_stats(p)
        assert (n, total, mx, mn) == (1, 10, 10, 10)

    def test_multiple_records(self, tmp_path):
        p = _write_fasta(tmp_path,
                         ">a\nACDE\n>b\nACDEFG\n>c\nAC\n")
        n, total, mx, mn = _fasta_stats(p)
        assert n == 3
        assert total == 12
        assert mx == 6
        assert mn == 2

    def test_multiline_sequence(self, tmp_path):
        p = _write_fasta(tmp_path,
                         ">long\nACDEFG\nHIKLMN\nPQRSTV\n")
        n, total, mx, mn = _fasta_stats(p)
        assert (n, total, mx, mn) == (1, 18, 18, 18)

    def test_blank_lines_ignored(self, tmp_path):
        p = _write_fasta(tmp_path,
                         ">a\nACDE\n\n>b\n\nGHIK\n")
        n, total, mx, mn = _fasta_stats(p)
        assert n == 2
        assert total == 8

    def test_empty_file_raises(self, tmp_path):
        p = _write_fasta(tmp_path, "")
        with pytest.raises(click.BadParameter, match="no FASTA records"):
            _fasta_stats(p)

    def test_no_headers_raises(self, tmp_path):
        p = _write_fasta(tmp_path, "ACDEFGHIKLMN\n")
        with pytest.raises(click.BadParameter, match="no FASTA records"):
            _fasta_stats(p)

    def test_headers_with_no_content_raises(self, tmp_path):
        p = _write_fasta(tmp_path, ">a\n>b\n>c\n")
        with pytest.raises(click.BadParameter, match="zero sequence content"):
            _fasta_stats(p)


class TestEstimateEmbeddingBytes:
    def test_mean_only_known_model(self):
        # esm2_t12_35M_UR50D: embed_dim=480
        # 1000 seqs × 480 × 1 layer × 4 bytes = 1,920,000 = ~1.9 MB
        b = _estimate_embedding_bytes(
            n_seqs=1000, total_len=300_000,
            model='esm2_t12_35M_UR50D', n_layers=1,
            include_mean=True, include_per_tok=False,
        )
        assert b == 1000 * 480 * 1 * 4

    def test_per_tok_only(self):
        # per-tok scales with total_len, not n_seqs
        b = _estimate_embedding_bytes(
            n_seqs=1000, total_len=300_000,
            model='esm2_t12_35M_UR50D', n_layers=1,
            include_mean=False, include_per_tok=True,
        )
        assert b == 300_000 * 480 * 1 * 4

    def test_both_mean_and_per_tok(self):
        b = _estimate_embedding_bytes(
            n_seqs=10, total_len=3000,
            model='esm2_t12_35M_UR50D', n_layers=1,
            include_mean=True, include_per_tok=True,
        )
        assert b == (10 * 480 * 4) + (3000 * 480 * 4)

    def test_multi_layer_scales(self):
        b1 = _estimate_embedding_bytes(
            100, 10_000, 'esm2_t12_35M_UR50D', 1, True, False,
        )
        b3 = _estimate_embedding_bytes(
            100, 10_000, 'esm2_t12_35M_UR50D', 3, True, False,
        )
        assert b3 == 3 * b1

    def test_unknown_model_returns_none(self):
        b = _estimate_embedding_bytes(
            100, 10_000, 'custom_finetune', 1, True, False,
        )
        assert b is None

    def test_nothing_requested_is_zero(self):
        b = _estimate_embedding_bytes(
            100, 10_000, 'esm2_t12_35M_UR50D', 1, False, False,
        )
        assert b == 0


class TestHumanizeBytes:
    def test_bytes(self):
        assert _humanize_bytes(512) == "512.0 B"

    def test_kilobytes(self):
        assert _humanize_bytes(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _humanize_bytes(5 * 1024 ** 2) == "5.0 MB"

    def test_gigabytes(self):
        assert _humanize_bytes(int(1.5 * 1024 ** 3)) == "1.5 GB"

    def test_terabytes_saturates(self):
        out = _humanize_bytes(3 * 1024 ** 4)
        assert out.endswith("TB")

    def test_petabytes_still_reports_as_tb(self):
        # With only TB as the top unit, values above saturate there.
        out = _humanize_bytes(5 * 1024 ** 5)
        assert out.endswith("TB")
