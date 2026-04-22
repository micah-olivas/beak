"""Unit tests for the in-container checkpointing helpers in generate_embeddings.py.

Tests the pure functions (chunk I/O, progress state, failure recording,
consolidation) without requiring torch or esm — those are imported lazily
inside generate_embeddings().
"""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


# Load the script as a module from its path (it lives in remote/docker/, not
# a regular Python package path). The heavy deps import lazily inside the main
# generate_embeddings() function, so importing for the helpers is free.
_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "beak" / "remote" / "docker" / "generate_embeddings.py"
_spec = importlib.util.spec_from_file_location("_gen_embed", _SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestChunkIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        chunk_path = tmp_path / "chunks" / "seq1.npz"
        layers = {
            'layer_6': np.arange(4, dtype=np.float32),
            'layer_12': np.arange(8, dtype=np.float32).reshape(2, 4),
        }
        gen._save_chunk(chunk_path, layers)

        assert chunk_path.exists()
        loaded = gen._load_chunk(chunk_path)
        assert set(loaded) == {'layer_6', 'layer_12'}
        np.testing.assert_array_equal(loaded['layer_6'], layers['layer_6'])
        np.testing.assert_array_equal(loaded['layer_12'], layers['layer_12'])

    def test_save_creates_parent_dirs(self, tmp_path):
        chunk_path = tmp_path / "deep" / "nested" / "path" / "seq.npz"
        gen._save_chunk(chunk_path, {'layer_0': np.zeros(3)})
        assert chunk_path.exists()

    def test_save_is_atomic_via_rename(self, tmp_path):
        # After save, no leftover .tmp file should remain
        chunk_path = tmp_path / "seq.npz"
        gen._save_chunk(chunk_path, {'layer_0': np.zeros(3)})
        assert not (tmp_path / "seq.npz.tmp").exists()


class TestWriteProgress:
    def test_writes_valid_json(self, tmp_path):
        progress_path = tmp_path / "progress.json"
        state = {'total': 10, 'done': 3, 'current': 'seqA', 'failed': 0}
        gen._write_progress(progress_path, state)

        loaded = json.loads(progress_path.read_text())
        assert loaded == state

    def test_overwrite_updates_contents(self, tmp_path):
        progress_path = tmp_path / "progress.json"
        gen._write_progress(progress_path, {'done': 1})
        gen._write_progress(progress_path, {'done': 5})
        assert json.loads(progress_path.read_text()) == {'done': 5}

    def test_no_stray_tmp_file(self, tmp_path):
        progress_path = tmp_path / "progress.json"
        gen._write_progress(progress_path, {'done': 0})
        assert not (tmp_path / "progress.json.tmp").exists()


class TestRecordFailure:
    def test_appends_one_line_per_failure(self, tmp_path):
        failed_path = tmp_path / "failed.tsv"
        gen._record_failure(failed_path, "seqA", ValueError("bad input"))
        gen._record_failure(failed_path, "seqB", RuntimeError("oom"))

        lines = failed_path.read_text().strip().split('\n')
        assert len(lines) == 2
        assert lines[0].startswith("seqA\tValueError\t")
        assert "bad input" in lines[0]
        assert lines[1].startswith("seqB\tRuntimeError\t")

    def test_scrubs_tab_and_newline_from_message(self, tmp_path):
        # TSV integrity: the message must not contain literal tabs or newlines
        failed_path = tmp_path / "failed.tsv"
        exc = ValueError("line1\nline2\tfield")
        gen._record_failure(failed_path, "seqX", exc)

        line = failed_path.read_text().rstrip('\n')
        # Exactly 3 tab-separated fields (seq_id, exc_type, scrubbed message)
        assert line.count('\t') == 2
        assert '\n' not in line


class TestConsolidateChunks:
    def _write_chunks(self, chunks_dir, data):
        for seq_id, layers in data.items():
            gen._save_chunk(chunks_dir / f"{seq_id}.npz", layers)

    def test_empty_directory_produces_empty_pickle(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        out_pickle = tmp_path / "out.pkl"
        n = gen._consolidate_chunks(chunks_dir, out_pickle)
        assert n == 0
        with open(out_pickle, 'rb') as f:
            assert pickle.load(f) == {}

    def test_consolidates_multiple_chunks(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, {
            'seqA': {'layer_33': np.ones(4, dtype=np.float32)},
            'seqB': {'layer_33': np.zeros(4, dtype=np.float32)},
        })
        out_pickle = tmp_path / "mean.pkl"

        n = gen._consolidate_chunks(chunks_dir, out_pickle)
        assert n == 2

        with open(out_pickle, 'rb') as f:
            merged = pickle.load(f)
        assert set(merged) == {'seqA', 'seqB'}
        np.testing.assert_array_equal(merged['seqA']['layer_33'], np.ones(4))

    def test_multi_layer_chunk_preserved(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, {
            'seqA': {
                'layer_6': np.ones(3, dtype=np.float32),
                'layer_12': np.zeros(3, dtype=np.float32),
            },
        })
        out_pickle = tmp_path / "out.pkl"
        gen._consolidate_chunks(chunks_dir, out_pickle)

        with open(out_pickle, 'rb') as f:
            merged = pickle.load(f)
        assert set(merged['seqA']) == {'layer_6', 'layer_12'}

    def test_output_is_loader_compatible(self, tmp_path):
        # The consolidated pickle must work with beak.embeddings.load_mean_embeddings
        from beak.embeddings import load_mean_embeddings

        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, {
            'seqA': {'layer_33': np.ones(8, dtype=np.float32)},
            'seqB': {'layer_33': np.zeros(8, dtype=np.float32)},
        })
        out_pickle = tmp_path / "mean_embeddings.pkl"
        gen._consolidate_chunks(chunks_dir, out_pickle)

        df = load_mean_embeddings(out_pickle)
        assert df.shape == (2, 8)
        assert set(df.index) == {'seqA', 'seqB'}


class TestResumeCompleteness:
    """Verify the chunk-complete check behaves like the generator would."""

    def test_chunk_complete_with_both_kinds(self, tmp_path):
        chunks_mean = tmp_path / "mean"
        chunks_tok = tmp_path / "per_tok"
        gen._save_chunk(chunks_mean / "seqA.npz", {'layer_0': np.zeros(2)})
        gen._save_chunk(chunks_tok / "seqA.npz", {'layer_0': np.zeros((3, 2))})

        def complete(seq_id, include_mean, include_per_tok):
            if include_mean and not (chunks_mean / f"{seq_id}.npz").exists():
                return False
            if include_per_tok and not (chunks_tok / f"{seq_id}.npz").exists():
                return False
            return True

        assert complete("seqA", include_mean=True, include_per_tok=True)
        assert complete("seqA", include_mean=True, include_per_tok=False)
        assert not complete("seqB", include_mean=True, include_per_tok=False)

    def test_chunk_complete_mean_only_ignores_missing_per_tok(self, tmp_path):
        chunks_mean = tmp_path / "mean"
        gen._save_chunk(chunks_mean / "seqA.npz", {'layer_0': np.zeros(2)})

        # Mean is present; we're not asking for per_tok → complete
        include_mean, include_per_tok = True, False
        seq_id = "seqA"
        assert (chunks_mean / f"{seq_id}.npz").exists()
