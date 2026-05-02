"""Build per-homolog organism-trait table from metaTraits.

Cheap once the dataset is cached on disk: load taxonomy.parquet, run
`annotate_traits`, save to `homologs/<active set>/traits.parquet`.

Triggered by the layers panel after taxonomy lands. Returns silently
when metaTraits isn't reachable so the rest of the project flow keeps
working offline.
"""

from typing import Optional

import pandas as pd


def build_traits_table(project, force: bool = False) -> Optional[pd.DataFrame]:
    """Materialise `homologs/<set>/traits.parquet` for a project.

    Reads the existing taxonomy.parquet for the active set, joins
    metaTraits, and saves the result. Returns the merged DataFrame, or
    None when there's no taxonomy yet or metaTraits isn't available.
    """
    homologs_dir = project.active_homologs_dir()
    tax_path = homologs_dir / "taxonomy.parquet"
    if not tax_path.exists():
        return None

    cache = homologs_dir / "traits.parquet"
    if cache.exists() and not force:
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass  # bad cache; recompute

    try:
        tax = pd.read_parquet(tax_path)
    except Exception:
        return None
    if tax.empty:
        return None

    from ..datasets import load_metatraits
    traits = load_metatraits()
    if traits is None or traits.empty:
        return None

    from ..traits import annotate_traits
    merged = annotate_traits(
        tax, organism_col="organism", taxon_id_col="taxon_id", traits=traits
    )

    try:
        merged.to_parquet(cache, index=False)
    except Exception:
        pass
    return merged
