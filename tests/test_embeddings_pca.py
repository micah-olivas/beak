"""Tests for the PCA helpers in beak.embeddings (notebook quickplots)."""

import matplotlib

matplotlib.use('Agg')  # headless, no display required
import numpy as np
import pandas as pd
import pytest

from beak.embeddings import pca, plot_pca


def _synthetic_embeddings(n_seqs=100, dim=32, seed=0):
    """Anisotropic Gaussian with a clear top principal direction."""
    rng = np.random.default_rng(seed)
    # Signal along axis 0, low noise elsewhere
    sig = rng.standard_normal(n_seqs) * 5.0
    noise = rng.standard_normal((n_seqs, dim)) * 0.1
    X = noise.copy()
    X[:, 0] += sig
    ids = [f"seq_{i:03d}" for i in range(n_seqs)]
    return pd.DataFrame(X, index=pd.Index(ids, name='seq_id'),
                        columns=[f"dim_{i}" for i in range(dim)])


class TestPca:
    def test_returns_expected_shape(self):
        df = _synthetic_embeddings(50, 16)
        pcs = pca(df, n_components=3)
        assert pcs.shape == (50, 3)
        assert list(pcs.columns) == ['PC1', 'PC2', 'PC3']

    def test_index_preserved(self):
        df = _synthetic_embeddings(20, 8)
        pcs = pca(df, n_components=2)
        assert list(pcs.index) == list(df.index)
        assert pcs.index.name == 'seq_id'

    def test_first_component_captures_dominant_axis(self):
        # With signal injected only on dim_0, PC1 should explain
        # well over half of the variance.
        df = _synthetic_embeddings(100, 32)
        pcs = pca(df, n_components=2)
        ratios = pcs.attrs['explained_variance_ratio']
        assert ratios[0] > 0.9

    def test_ratios_sum_to_at_most_one(self):
        df = _synthetic_embeddings(50, 16)
        pcs = pca(df, n_components=16)
        ratios = pcs.attrs['explained_variance_ratio']
        assert 0.999 <= sum(ratios) <= 1.0001

    def test_clamps_to_min_dim(self):
        df = _synthetic_embeddings(5, 4)
        pcs = pca(df, n_components=10)  # more than features or samples
        assert pcs.shape[1] <= min(5, 4)

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame()
        pcs = pca(df)
        assert pcs.empty

    def test_doesnt_mutate_input(self):
        df = _synthetic_embeddings(10, 8)
        snapshot = df.copy()
        _ = pca(df, n_components=2)
        pd.testing.assert_frame_equal(df, snapshot)


class TestPlotPca:
    def test_returns_fig_ax_pcs(self):
        df = _synthetic_embeddings(20, 8)
        fig, ax, pcs = plot_pca(df)
        assert fig is not None
        assert ax is not None
        assert pcs.shape == (20, 2)

    def test_accepts_precomputed_pcs(self):
        df = _synthetic_embeddings(20, 8)
        pcs = pca(df, n_components=2)
        fig, ax, out = plot_pca(pcs)
        # Should reuse the supplied DataFrame rather than recomputing.
        # The returned object is the same identity as the input.
        assert out is pcs

    def test_axis_labels_include_variance(self):
        df = _synthetic_embeddings(30, 8)
        _, ax, _ = plot_pca(df)
        assert '%' in ax.get_xlabel()
        assert '%' in ax.get_ylabel()

    def test_numeric_color_draws_colorbar(self):
        df = _synthetic_embeddings(30, 8)
        length = pd.Series(np.arange(30), index=df.index, name='length')
        fig, ax, _ = plot_pca(df, color=length)
        # Numeric -> colorbar present (extra axes attached to fig)
        assert len(fig.axes) >= 2

    def test_categorical_color_draws_legend(self):
        df = _synthetic_embeddings(30, 8)
        groups = pd.Series(['A', 'B', 'C'] * 10, index=df.index, name='group')
        fig, ax, _ = plot_pca(df, color=groups)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert set(labels) == {'A', 'B', 'C'}

    def test_color_by_column_name(self):
        df = _synthetic_embeddings(30, 8)
        # Treat one of the dim columns as a colour source
        fig, ax, _ = plot_pca(df, color='dim_0')
        assert len(fig.axes) >= 2  # colorbar added

    def test_unknown_column_raises(self):
        df = _synthetic_embeddings(10, 4)
        with pytest.raises(KeyError, match="not a column"):
            plot_pca(df, color='does_not_exist')

    def test_length_mismatch_raises(self):
        df = _synthetic_embeddings(10, 4)
        bad = [1, 2, 3]  # wrong length
        with pytest.raises(ValueError, match="length"):
            plot_pca(df, color=bad)

    def test_plot_title_shows_n_sequences(self):
        df = _synthetic_embeddings(42, 8)
        _, ax, _ = plot_pca(df)
        assert '42' in ax.get_title()
