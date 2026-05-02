"""Join organism-level metaTraits onto a per-homolog taxonomy frame.

metaTraits (EMBL 2026) consolidates BacDive, BV-BRC, JGI IMG, and GOLD
into ~140 harmonised organism trait columns across 2.2M genomes. We
treat it like the Enqvist growth-temp dataset: pull, cache, join.

The matching cascade per row is:
    1. taxon_id (NCBI) — the cleanest signal when present
    2. scientific name (case-insensitive, exact)
    3. genus name (the species name's first whitespace-delimited token) —
       a fallback when species isn't represented but its genus is

Trait columns are returned with a `trait_` prefix so they're easy to
spot when a downstream caller reaches for them.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


_PREFIX = "trait_"

# Lineage / identifier columns we shouldn't propagate into the merged
# frame — those would clobber the caller's existing taxonomy columns.
_NON_TRAIT_COLS = {
    "taxon_id", "tax_id", "ncbi_taxon_id", "gtdb_taxon_id",
    "scientific_name", "organism", "name",
    "domain", "superkingdom", "phylum", "class", "order",
    "family", "genus", "species",
    "lineage", "lineage_text",
}


def _trait_columns(traits: pd.DataFrame) -> List[str]:
    return [c for c in traits.columns if c not in _NON_TRAIT_COLS]


def _normalize(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _genus_of(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return None
    return name.split()[0].lower() or None


def _species_of(name: Optional[str]) -> Optional[str]:
    """Genus + species, lowered. ``"Escherichia coli K-12"`` → ``"escherichia coli"``."""
    if not isinstance(name, str) or not name:
        return None
    parts = name.split()
    if len(parts) < 2:
        return None
    return f"{parts[0]} {parts[1]}".lower()


def annotate_traits(
    df: pd.DataFrame,
    organism_col: str = "organism",
    taxon_id_col: Optional[str] = "taxon_id",
    traits: Optional[pd.DataFrame] = None,
    columns: Optional[Iterable[str]] = None,
    prefix: str = _PREFIX,
) -> pd.DataFrame:
    """Attach metaTraits to each row of ``df``.

    Args:
        df: Per-sequence frame with at least ``organism`` (or whatever
            ``organism_col`` resolves to). ``taxon_id_col`` is consulted
            first when present.
        organism_col: Name of the scientific-name column in ``df``.
        taxon_id_col: Name of the NCBI tax-id column in ``df`` (or None
            to skip the tax-id pass).
        traits: Optional pre-loaded metaTraits frame. Default behaviour
            calls ``datasets.load_metatraits()``.
        columns: If given, restrict the join to these trait columns.
        prefix: Prefix applied to merged trait columns (default
            ``trait_``).

    Returns:
        ``df`` with new ``{prefix}<col>`` columns appended, plus a
        ``trait_match_level`` column ("taxon_id" / "scientific_name" /
        "genus" / None) describing how each row was matched. If the
        metaTraits dataset is unavailable, returns ``df`` unchanged.
    """
    out = df.copy()
    if traits is None:
        from .datasets import load_metatraits
        traits = load_metatraits()
    if traits is None or traits.empty:
        return out

    trait_cols = _trait_columns(traits)
    if columns is not None:
        wanted = set(columns)
        trait_cols = [c for c in trait_cols if c in wanted]
    if not trait_cols:
        return out

    # Each lookup keys map to a row index in `traits`.
    by_taxid: dict = {}
    if taxon_id_col and "taxon_id" in traits.columns:
        for idx, tid in traits["taxon_id"].items():
            if pd.notna(tid):
                by_taxid[int(tid)] = idx

    name_col = "scientific_name" if "scientific_name" in traits.columns else (
        "organism" if "organism" in traits.columns else None
    )
    by_name: dict = {}
    by_species: dict = {}
    by_genus: dict = {}
    if name_col is not None:
        names = _normalize(traits[name_col])
        for idx, nm in names.items():
            if not nm or pd.isna(nm):
                continue
            by_name.setdefault(nm, idx)
            parts = nm.split()
            if len(parts) >= 2:
                by_species.setdefault(f"{parts[0]} {parts[1]}", idx)
            by_genus.setdefault(parts[0], idx)

    # Resolve a row index per output row.
    match_idx: List[Optional[int]] = []
    match_lvl: List[Optional[str]] = []
    for _, row in out.iterrows():
        idx: Optional[int] = None
        lvl: Optional[str] = None
        if taxon_id_col and taxon_id_col in out.columns and by_taxid:
            tid = row.get(taxon_id_col)
            if pd.notna(tid):
                try:
                    idx = by_taxid.get(int(tid))
                    if idx is not None:
                        lvl = "taxon_id"
                except (TypeError, ValueError):
                    pass
        if idx is None and organism_col in out.columns and by_name:
            org = row.get(organism_col)
            if isinstance(org, str) and org.strip():
                key = org.strip().lower()
                idx = by_name.get(key)
                if idx is not None:
                    lvl = "scientific_name"
                else:
                    sp = _species_of(org)
                    if sp:
                        idx = by_species.get(sp)
                        if idx is not None:
                            lvl = "species"
                    if idx is None:
                        g = _genus_of(org)
                        if g:
                            idx = by_genus.get(g)
                            if idx is not None:
                                lvl = "genus"
        match_idx.append(idx)
        match_lvl.append(lvl)

    # Pull the trait values via reindex; rows without a match get NaN.
    sub = traits.loc[:, trait_cols].reset_index(drop=True)
    pulled = sub.reindex(match_idx).reset_index(drop=True)
    pulled.columns = [f"{prefix}{c}" for c in pulled.columns]
    pulled.index = out.index
    out = pd.concat([out, pulled], axis=1)
    out["trait_match_level"] = match_lvl
    return out
