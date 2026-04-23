"""Loaders for ESM embedding pickles produced by `beak embeddings`.

The pickles are produced by the in-container `generate_embeddings.py` with
this shape:

    mean_embeddings.pkl:       {seq_id: {layer_N: np.ndarray(embed_dim,)}}
    per_token_embeddings.pkl:  {seq_id: {layer_N: np.ndarray(seq_len, embed_dim)}}

These loaders turn them into pandas DataFrames that downstream analysis
code can join to sequence metadata or residue tables directly.
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def _load_pickle(path: Union[str, Path]) -> dict:
    """Load an embedding pickle and return the raw dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")
    with open(p, 'rb') as f:
        return pickle.load(f)


def _pick_layer(sample: dict, layer: Optional[Union[int, str]]) -> str:
    """Resolve a layer selector against the available layer keys.

    Layer keys in the pickle look like 'layer_33'. The selector may be:
      - None:      auto-pick if there is exactly one layer, else raise.
      - int:       matched against the trailing number ('layer_N').
      - str:       matched against the full key.
    """
    available = list(sample.keys())
    if not available:
        raise ValueError("Embedding entry has no layers")

    if layer is None:
        if len(available) == 1:
            return available[0]
        raise ValueError(
            f"Multiple layers present ({available}); pass layer=... to pick one."
        )

    if isinstance(layer, int):
        key = f"layer_{layer}"
        if key in available:
            return key
        raise ValueError(
            f"Layer {layer} not in pickle (layers: {available})"
        )

    if layer in available:
        return layer
    raise ValueError(f"Layer '{layer}' not in pickle (layers: {available})")


def load_mean_embeddings(
    path: Union[str, Path],
    layer: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Load a mean-pooled embeddings pickle into a flat DataFrame.

    Args:
        path: path to mean_embeddings.pkl
        layer: which layer to extract. If None, auto-picks when exactly one
            layer is present; otherwise raises.

    Returns:
        DataFrame with index=seq_id and columns=[dim_0, dim_1, ..., dim_{D-1}].
        Each row is the per-sequence embedding vector for the chosen layer.
    """
    raw = _load_pickle(path)
    if not raw:
        return pd.DataFrame()

    sample_layers = next(iter(raw.values()))
    layer_key = _pick_layer(sample_layers, layer)

    records = {seq_id: layers[layer_key] for seq_id, layers in raw.items()}
    df = pd.DataFrame.from_dict(records, orient='index')
    df.columns = [f"dim_{i}" for i in range(df.shape[1])]
    df.index.name = 'seq_id'
    return df


def load_per_token_embeddings(
    path: Union[str, Path],
    layer: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Load a per-token embeddings pickle into a long-form DataFrame.

    Args:
        path: path to per_token_embeddings.pkl
        layer: which layer to extract (see load_mean_embeddings).

    Returns:
        DataFrame with a MultiIndex of (seq_id, position) and columns
        [dim_0, ..., dim_{D-1}]. position is 1-based and matches the residue
        numbering of the input FASTA sequence (special tokens stripped).
    """
    raw = _load_pickle(path)
    if not raw:
        return pd.DataFrame()

    sample_layers = next(iter(raw.values()))
    layer_key = _pick_layer(sample_layers, layer)

    frames = []
    for seq_id, layers in raw.items():
        arr = layers[layer_key]  # shape (seq_len, embed_dim)
        n_positions, embed_dim = arr.shape
        frame = pd.DataFrame(
            arr,
            index=pd.MultiIndex.from_product(
                [[seq_id], np.arange(1, n_positions + 1)],
                names=['seq_id', 'position'],
            ),
            columns=[f"dim_{i}" for i in range(embed_dim)],
        )
        frames.append(frame)

    return pd.concat(frames) if frames else pd.DataFrame()


def pca(df: pd.DataFrame, n_components: int = 2, random_state=None) -> pd.DataFrame:
    """Principal-component transform of a wide embedding DataFrame.

    Thin wrapper around ``sklearn.decomposition.PCA`` that returns a
    DataFrame aligned to the input index (seq_id for the output of
    ``load_mean_embeddings``; MultiIndex for per-token) with columns
    ``PC1..PCk``. Explained variance is stashed on ``result.attrs`` so
    ``plot_pca`` can label axes without recomputing.

    Args:
        df: embedding matrix. Must be numeric (no metadata columns).
        n_components: number of components to return. Clamped to
            ``min(n_samples, n_features)`` by sklearn.
        random_state: forwarded to ``PCA`` for deterministic results
            when sklearn picks the randomized solver.

    Returns:
        DataFrame of shape (n_samples, n_components) with columns
        ['PC1', 'PC2', ...]. ``result.attrs['explained_variance_ratio']``
        is the per-component fraction of total variance captured.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    from sklearn.decomposition import PCA

    n_samples, n_features = df.shape
    k = min(n_components, n_samples, n_features)

    model = PCA(n_components=k, random_state=random_state)
    scores = model.fit_transform(df.values)

    result = pd.DataFrame(
        scores,
        index=df.index,
        columns=[f"PC{i + 1}" for i in range(k)],
    )
    result.attrs['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
    return result


def plot_pca(
    df: pd.DataFrame,
    color=None,
    n_components: int = 2,
    cmap: str = 'viridis',
    figsize=(7, 6),
    ax=None,
    **scatter_kwargs,
):
    """Quick scatter of the first two principal components.

    Accepts either a raw embedding DataFrame (PCA is computed on the
    fly) or a pre-computed PC DataFrame from `pca()` (detected by
    columns starting with 'PC'). Intended for fast notebook inspection;
    not a replacement for a proper UMAP/t-SNE workflow.

    Args:
        df: embedding matrix or output of `pca()`.
        color: optional per-sample colouring. Accepts a pandas Series
            (aligned to df.index), an array-like of length n_samples,
            or a column name in `df`. Numeric values produce a colour
            bar; non-numeric (categorical) produces a legend with one
            scatter call per category.
        n_components: how many PCs to compute (plot always uses the
            first two; more are available via the returned DataFrame).
        cmap: matplotlib colormap for numeric colouring.
        figsize: figure size if a new axis is created.
        ax: reuse an existing Matplotlib Axes. If None, a new figure
            is created.
        **scatter_kwargs: forwarded to `ax.scatter` (e.g. `s=5, alpha=0.7`).

    Returns:
        (fig, ax, pcs) — the matplotlib Figure, Axes, and the PC
        DataFrame so the caller can reuse it or inspect
        explained_variance_ratio.
    """
    import matplotlib.pyplot as plt

    # Accept pre-computed PCs to avoid recomputing SVD
    is_pcs = any(str(c).startswith('PC') for c in df.columns)
    pcs = df if is_pcs else pca(df, n_components=max(n_components, 2))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Resolve the colour series
    color_series = None
    color_name = None
    if color is not None:
        if isinstance(color, str):
            if color not in df.columns:
                raise KeyError(
                    f"color={color!r} is not a column in df. "
                    f"Pass a Series / array-like or a valid column name."
                )
            color_series = df[color]
            color_name = color
        elif isinstance(color, pd.Series):
            color_series = color
            color_name = color.name
        else:
            arr = np.asarray(color)
            if len(arr) != len(pcs):
                raise ValueError(
                    f"color has length {len(arr)} but pcs has {len(pcs)} rows."
                )
            color_series = pd.Series(arr, index=pcs.index)

    defaults = {'s': 14, 'alpha': 0.75, 'edgecolor': 'none'}
    defaults.update(scatter_kwargs)

    if color_series is None:
        ax.scatter(pcs['PC1'], pcs['PC2'], **defaults)
    elif pd.api.types.is_numeric_dtype(color_series):
        sc = ax.scatter(pcs['PC1'], pcs['PC2'],
                        c=color_series.values, cmap=cmap, **defaults)
        fig.colorbar(sc, ax=ax, label=color_name or '')
    else:
        # Categorical: one scatter call per group so we get a legend
        for category, mask in color_series.groupby(color_series, observed=True).groups.items():
            subset = pcs.loc[mask]
            ax.scatter(subset['PC1'], subset['PC2'],
                       label=str(category), **defaults)
        ax.legend(title=color_name or None,
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=False)

    # Labels with explained variance where available
    explained = pcs.attrs.get('explained_variance_ratio')
    xlab, ylab = 'PC1', 'PC2'
    if explained and len(explained) >= 2:
        xlab = f"PC1 ({explained[0] * 100:.1f}%)"
        ylab = f"PC2 ({explained[1] * 100:.1f}%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{len(pcs):,} sequences · PCA")

    return fig, ax, pcs


def load_embeddings(
    path: Union[str, Path],
    layer: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Load either a mean or per-token embedding pickle, auto-detecting.

    Dispatches by inspecting the first entry's shape:
      - 1D array -> mean-pooled layout -> load_mean_embeddings
      - 2D array -> per-token layout   -> load_per_token_embeddings

    Falls back to filename sniffing if the dict is empty.
    """
    raw = _load_pickle(path)

    if raw:
        sample_layers = next(iter(raw.values()))
        sample_array = next(iter(sample_layers.values()))
        ndim = np.asarray(sample_array).ndim
        if ndim == 1:
            return load_mean_embeddings(path, layer=layer)
        if ndim == 2:
            return load_per_token_embeddings(path, layer=layer)
        raise ValueError(f"Unexpected embedding array dimensionality: {ndim}")

    # Empty pickle: fall back to filename
    name = Path(path).name.lower()
    if 'per_token' in name or 'per-tok' in name:
        return load_per_token_embeddings(path, layer=layer)
    return load_mean_embeddings(path, layer=layer)
