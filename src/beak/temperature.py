"""Growth-temperature annotation from the Enqvist dataset."""

import pandas as pd
from typing import Optional


def annotate_temperature(
    df: pd.DataFrame,
    organism_col: str = 'scientific_name',
    method: str = 'exact'
) -> pd.DataFrame:
    """
    Annotate sequences with growth temperatures from Enqvist dataset

    Args:
        df: DataFrame with organism names
        organism_col: Column containing organism names
        method: 'exact' or 'fuzzy' matching

    Returns:
        DataFrame with added 'growth_temp' and 'temp_source' columns

    Example:
        >>> tax_results = taxonomy.get_results(job_id)
        >>> annotated = annotate_temperature(tax_results)
        >>> annotated[['scientific_name', 'growth_temp']].head()
    """
    from .datasets import load_growth_temps

    temps = load_growth_temps()

    if method == 'exact':
        merged = df.merge(
            temps[['organism', 'temperature', 'domain']],
            left_on=organism_col,
            right_on='organism',
            how='left'
        )
        merged = merged.rename(columns={
            'temperature': 'growth_temp',
            'domain': 'temp_domain'
        })
        merged = merged.drop(columns=['organism'], errors='ignore')

    elif method == 'fuzzy':
        try:
            from fuzzywuzzy import process
        except ImportError:
            raise ImportError(
                "Fuzzy matching requires 'fuzzywuzzy'. "
                "Install with: pip install fuzzywuzzy python-Levenshtein"
            )

        def find_best_match(name):
            if pd.isna(name):
                return None, 0
            match = process.extractOne(name, temps['organism'].tolist())
            return match[0] if match else None, match[1] if match else 0

        matches = df[organism_col].apply(find_best_match)
        df['matched_organism'] = [m[0] for m in matches]
        df['match_score'] = [m[1] for m in matches]

        merged = df.merge(
            temps[['organism', 'temperature', 'domain']],
            left_on='matched_organism',
            right_on='organism',
            how='left'
        )
        merged = merged.rename(columns={
            'temperature': 'growth_temp',
            'domain': 'temp_domain'
        })
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'fuzzy'")

    return merged


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
