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
