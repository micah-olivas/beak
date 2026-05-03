"""Growth-temperature annotation from the Enqvist dataset."""

import re

import pandas as pd
from typing import Optional


# UniProt scientific names ("Escherichia coli (strain K12)") differ in
# casing, separators, and decoration from the Enqvist dataset's
# normalized form ("escherichia_coli"). Both transforms below produce
# the Enqvist style, so a plain merge can find the row.
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_WS_RE = re.compile(r"\s+")


def _to_genus_species(name) -> Optional[str]:
    """Return ``"genus_species"`` from any UniProt-style organism name.

    Strips strain qualifiers, lowercases, replaces whitespace with
    underscores, then keeps only the first two tokens — that's the
    canonical key used in the Enqvist dataset.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = _PAREN_RE.sub("", str(name)).strip().lower()
    if not s:
        return None
    s = _WS_RE.sub("_", s)
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else None


def _to_genus(name) -> Optional[str]:
    """Return just the genus token from a UniProt-style organism name."""
    gs = _to_genus_species(name)
    if not gs:
        return None
    return gs.split("_", 1)[0]


def annotate_temperature(
    df: pd.DataFrame,
    organism_col: str = 'scientific_name',
    method: str = 'exact'
) -> pd.DataFrame:
    """
    Annotate sequences with growth temperatures from the Enqvist dataset.

    Two-stage match: first an exact ``genus_species`` lookup against the
    Enqvist organism column, then a genus-level fallback (median temp
    across species in the same genus) for any rows still unresolved.
    The `temp_source` column reports which path each row took
    (``'species'``, ``'genus'``, or ``NaN``) so downstream analysis can
    weight species-level matches more heavily.

    Args:
        df: DataFrame with organism names.
        organism_col: Column containing organism names.
        method: ``'exact'`` (default) or ``'fuzzy'`` (requires fuzzywuzzy).

    Returns:
        DataFrame with added ``growth_temp`` (°C) and ``temp_source`` columns.

    Example:
        >>> annotated = annotate_temperature(tax_df, organism_col="organism")
        >>> annotated[["organism", "growth_temp", "temp_source"]].head()
    """
    from .datasets import load_growth_temps

    temps = load_growth_temps()
    if organism_col not in df.columns:
        raise KeyError(
            f"organism_col {organism_col!r} not in DataFrame columns "
            f"({list(df.columns)})"
        )

    out = df.copy()
    if method == 'exact':
        # Build normalized join keys: genus_species and genus alone.
        # We do this on a copy so the original DataFrame isn't mutated.
        out['_organism_key'] = out[organism_col].map(_to_genus_species)
        out['_genus_key'] = out[organism_col].map(_to_genus)
    elif method == 'fuzzy':
        try:
            from fuzzywuzzy import process
        except ImportError:
            raise ImportError(
                "Fuzzy matching requires 'fuzzywuzzy'. "
                "Install with: pip install fuzzywuzzy python-Levenshtein"
            )
        choices = temps['organism'].tolist()

        def best(name):
            if pd.isna(name):
                return None
            normalized = _to_genus_species(name)
            if normalized is None:
                return None
            match = process.extractOne(normalized, choices)
            return match[0] if match else None

        out['_organism_key'] = out[organism_col].map(best)
        out['_genus_key'] = out[organism_col].map(_to_genus)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'fuzzy'")

    # Stage 1 — species-level match. Rename `organism` upfront so it
    # can't collide with a same-named column already on `df` (which is
    # exactly the case when the input is the project's taxonomy table).
    species_lookup = (
        temps[['organism', 'temperature']]
        .rename(columns={'organism': '_organism_key', 'temperature': 'growth_temp'})
    )
    out = out.merge(species_lookup, on='_organism_key', how='left')
    out['temp_source'] = out['growth_temp'].notna().map(
        {True: 'species', False: pd.NA}
    )

    # Stage 2 — genus-level fallback: median temp across all species in
    # the genus. Median (not mean) so a single hyperthermophile in an
    # otherwise mesophilic genus doesn't drag the fallback to 80 °C.
    needs_fallback = out['growth_temp'].isna() & out['_genus_key'].notna()
    if needs_fallback.any():
        genus_temps = (
            temps[['organism', 'temperature']]
            .assign(_genus_key=lambda d: d['organism'].str.split('_').str[0])
            .groupby('_genus_key', as_index=False)['temperature']
            .median()
            .rename(columns={'temperature': '_genus_temp'})
        )
        out = out.merge(genus_temps, on='_genus_key', how='left')
        # Only fill rows that don't already have a species-level match;
        # `combine_first` would clobber NaN sources with the genus value
        # too aggressively here.
        fill_mask = out['growth_temp'].isna() & out['_genus_temp'].notna()
        out.loc[fill_mask, 'growth_temp'] = out.loc[fill_mask, '_genus_temp']
        out.loc[fill_mask, 'temp_source'] = 'genus'
        out = out.drop(columns=['_genus_temp'])

    out = out.drop(columns=['_organism_key', '_genus_key'], errors='ignore')
    return out


def filter_by_temperature(
    df: pd.DataFrame,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    temp_col: str = 'growth_temp'
) -> pd.DataFrame:
    """
    Filter sequences by growth temperature range

    Args:
        df: DataFrame with temperature annotations
        min_temp: Minimum temperature (inclusive)
        max_temp: Maximum temperature (inclusive)
        temp_col: Column containing temperatures

    Returns:
        Filtered DataFrame

    Example:
        >>> # Get only thermophiles (>50°C)
        >>> thermophiles = filter_by_temperature(df, min_temp=50)
    """
    filtered = df.copy()

    if min_temp is not None:
        filtered = filtered[filtered[temp_col] >= min_temp]

    if max_temp is not None:
        filtered = filtered[filtered[temp_col] <= max_temp]

    return filtered
