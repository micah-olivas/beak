"""Unit tests for beak.embeddings loader module."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beak.embeddings import (
    load_embeddings,
    load_mean_embeddings,
    load_per_token_embeddings,
)


EMBED_DIM = 16


def _write_mean_pickle(path: Path, seq_ids, layers=(33,), embed_dim=EMBED_DIM, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        sid: {f"layer_{L}": rng.standard_normal(embed_dim).astype(np.float32)
              for L in layers}
        for sid in seq_ids
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def _write_per_tok_pickle(path: Path, seq_id_to_len, layers=(33,), embed_dim=EMBED_DIM, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        sid: {f"layer_{L}": rng.standard_normal((length, embed_dim)).astype(np.float32)
              for L in layers}
        for sid, length in seq_id_to_len.items()
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class TestLoadMeanEmbeddings:
    def test_single_layer_returns_flat_matrix(self, tmp_path):
        path = tmp_path / "mean_embeddings.pkl"
        _write_mean_pickle(path, seq_ids=['a', 'b', 'c'])

        df = load_mean_embeddings(path)

        assert df.shape == (3, EMBED_DIM)
        assert df.index.name == 'seq_id'
        assert list(df.index) == ['a', 'b', 'c']
        assert list(df.columns) == [f"dim_{i}" for i in range(EMBED_DIM)]

    def test_multiple_layers_requires_selector(self, tmp_path):
        path = tmp_path / "mean.pkl"
        _write_mean_pickle(path, seq_ids=['a'], layers=(6, 12))

        with pytest.raises(ValueError, match="Multiple layers"):
            load_mean_embeddings(path)

    def test_layer_int_selector(self, tmp_path):
        path = tmp_path / "mean.pkl"
        _write_mean_pickle(path, seq_ids=['a'], layers=(6, 12))

        df6 = load_mean_embeddings(path, layer=6)
        df12 = load_mean_embeddings(path, layer=12)

        assert df6.shape == (1, EMBED_DIM)
        assert df12.shape == (1, EMBED_DIM)
        # Different layers should give different vectors
        assert not np.allclose(df6.iloc[0].values, df12.iloc[0].values)

    def test_layer_string_selector(self, tmp_path):
        path = tmp_path / "mean.pkl"
        _write_mean_pickle(path, seq_ids=['a'], layers=(33,))

        df = load_mean_embeddings(path, layer='layer_33')
        assert df.shape == (1, EMBED_DIM)

    def test_unknown_layer_raises(self, tmp_path):
        path = tmp_path / "mean.pkl"
        _write_mean_pickle(path, seq_ids=['a'], layers=(33,))

        with pytest.raises(ValueError, match="Layer 99"):
            load_mean_embeddings(path, layer=99)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mean_embeddings(tmp_path / "does_not_exist.pkl")

    def test_empty_pickle_returns_empty_df(self, tmp_path):
        path = tmp_path / "empty.pkl"
        with open(path, 'wb') as f:
            pickle.dump({}, f)
        df = load_mean_embeddings(path)
        assert df.empty


class TestLoadPerTokenEmbeddings:
    def test_long_form_with_multiindex(self, tmp_path):
        path = tmp_path / "per_token_embeddings.pkl"
        _write_per_tok_pickle(path, seq_id_to_len={'a': 5, 'b': 3})

        df = load_per_token_embeddings(path)

        assert df.shape == (5 + 3, EMBED_DIM)
        assert df.index.names == ['seq_id', 'position']
        # Positions are 1-based
        a_positions = df.loc['a'].index.tolist()
        assert a_positions == [1, 2, 3, 4, 5]
        b_positions = df.loc['b'].index.tolist()
        assert b_positions == [1, 2, 3]

    def test_columns_are_dim_labels(self, tmp_path):
        path = tmp_path / "pt.pkl"
        _write_per_tok_pickle(path, seq_id_to_len={'a': 4})
        df = load_per_token_embeddings(path)
        assert list(df.columns) == [f"dim_{i}" for i in range(EMBED_DIM)]

    def test_layer_selector_works(self, tmp_path):
        path = tmp_path / "pt.pkl"
        _write_per_tok_pickle(path, seq_id_to_len={'a': 4}, layers=(6, 12))
        df = load_per_token_embeddings(path, layer=12)
        assert df.shape == (4, EMBED_DIM)

    def test_single_position_sequence(self, tmp_path):
        path = tmp_path / "pt.pkl"
        _write_per_tok_pickle(path, seq_id_to_len={'a': 1})
        df = load_per_token_embeddings(path)
        assert df.shape == (1, EMBED_DIM)
        assert df.index.tolist() == [('a', 1)]


class TestLoadEmbeddingsDispatch:
    def test_auto_detects_mean_from_1d_array(self, tmp_path):
        path = tmp_path / "mean_embeddings.pkl"
        _write_mean_pickle(path, seq_ids=['a', 'b'])
        df = load_embeddings(path)
        # Mean layout: index is a flat seq_id
        assert df.index.name == 'seq_id'
        assert df.shape == (2, EMBED_DIM)

    def test_auto_detects_per_token_from_2d_array(self, tmp_path):
        path = tmp_path / "per_token_embeddings.pkl"
        _write_per_tok_pickle(path, seq_id_to_len={'a': 3})
        df = load_embeddings(path)
        assert df.index.names == ['seq_id', 'position']
        assert df.shape == (3, EMBED_DIM)

    def test_empty_pickle_falls_back_to_filename(self, tmp_path):
        mean_path = tmp_path / "mean_embeddings.pkl"
        tok_path = tmp_path / "per_token_embeddings.pkl"
        for p in (mean_path, tok_path):
            with open(p, 'wb') as f:
                pickle.dump({}, f)
        assert load_embeddings(mean_path).empty
        assert load_embeddings(tok_path).empty
