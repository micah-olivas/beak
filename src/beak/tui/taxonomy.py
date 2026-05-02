"""Build per-homolog taxonomy + growth-temperature table.

For each hit in `homologs/sequences.fasta` we:
1. Parse out the UniProt accession (works for `sp|P00533|...` and
   `UniRef90_P00533` style IDs).
2. Batch-query UniProt REST for organism + lineage.
3. Run `annotate_temperature` from beak.temperature to attach growth
   temps from the Enqvist dataset (genus-level fallback).
4. Save the result as `homologs/taxonomy.parquet`.

Cached on disk; the layers panel busts the cache when homologs are
re-pulled or reset.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_BATCH_SIZE = 50
_ACCESSION_RE = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")


def _accession_from_seq_id(seq_id: str) -> Optional[str]:
    """Pull a UniProt accession out of common FASTA id formats."""
    # Strip common prefixes (UniRef90_, UniRef100_, sp|, tr|).
    candidate = seq_id
    for pfx in ("UniRef90_", "UniRef100_", "UniRef50_", "sp|", "tr|"):
        if candidate.startswith(pfx):
            candidate = candidate[len(pfx):]
            break
    # Take the chunk before the next pipe / space if any.
    candidate = candidate.split("|", 1)[0].split(" ", 1)[0]
    m = _ACCESSION_RE.search(candidate)
    return m.group(1) if m else None


def _parse_fasta_ids(fasta_path: Path) -> List[str]:
    """Yield raw header tokens (first whitespace-separated chunk) per record."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                ids.append(header.split()[0] if header else "")
    return ids


# Taxonomic ranks we expose as standalone columns. UniProt returns the
# free-form ancestral lineage (`lineage`) plus a `lineages` array where
# each entry has a `rank`; we pull values out of the latter so each
# column is rank-correct rather than positional. Order here drives the
# default cycle in the PCA legend.
_RANK_LEVELS: List[str] = [
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


def _fetch_batch(accessions: List[str]) -> List[dict]:
    """Query UniProt for one batch of accessions; returns row dicts."""
    if not accessions:
        return []
    query = " OR ".join(f"accession:{a}" for a in accessions)
    params = {
        # `lineage_ids` returns rank-tagged ancestors so we can populate
        # superkingdom/kingdom/phylum/class/order/family/genus/species
        # columns without relying on positional guesses.
        "query": query,
        "fields": "accession,organism_name,organism_id,lineage,lineage_ids",
        "format": "json",
        "size": str(len(accessions)),
    }
    url = f"{_UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return []
    rows = []
    for r in data.get("results", []):
        org = r.get("organism") or {}
        lineage = org.get("lineage") or []
        # `lineages` (note the trailing 's') is the rank-tagged variant.
        # UniProt returns it under either `lineages` or `lineage_ids`
        # depending on the requested fields; check both.
        ranked = r.get("lineages") or r.get("lineageIds") or []
        rank_map: Dict[str, Optional[str]] = {}
        for entry in ranked:
            rank = (entry.get("rank") or "").lower()
            name = entry.get("scientificName")
            if rank in _RANK_LEVELS and name and rank not in rank_map:
                rank_map[rank] = name
        # Always include the species name from `organism.scientificName`
        # if the lineage didn't surface a species-rank entry — it's the
        # most precise label we have.
        if "species" not in rank_map and org.get("scientificName"):
            rank_map["species"] = org.get("scientificName")

        row = {
            "uniprot_id": r.get("primaryAccession"),
            "organism": org.get("scientificName"),
            "taxon_id": org.get("taxonId"),
            "lineage": lineage,
            # Legacy alias: callers still read `domain`. Map it to
            # superkingdom so existing code keeps working.
            "domain": rank_map.get("superkingdom")
                      or (lineage[0] if len(lineage) >= 1 else None),
        }
        for rank in _RANK_LEVELS:
            row[rank] = rank_map.get(rank)
        # Fallback for older UniProt responses that don't provide
        # `lineages` — populate phylum/class positionally so we never
        # regress when the rank-tagged path is unavailable.
        if row.get("phylum") is None and len(lineage) >= 2:
            row["phylum"] = lineage[1]
        if row.get("class") is None and len(lineage) >= 3:
            row["class"] = lineage[2]
        rows.append(row)
    return rows


def build_taxonomy_table(project, force: bool = False) -> Optional[pd.DataFrame]:
    """Materialise `homologs/taxonomy.parquet` for a project.

    Returns the DataFrame, or None if no homologs exist. Cached: returns
    the existing file if present unless `force=True`.
    """
    homologs_dir = project.active_homologs_dir()
    fasta = homologs_dir / "sequences.fasta"
    if not fasta.exists():
        return None

    cache = homologs_dir / "taxonomy.parquet"
    if cache.exists() and not force:
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass  # bad cache; recompute

    seq_ids = _parse_fasta_ids(fasta)
    if not seq_ids:
        return None

    # Map seq_id → accession (drops any we can't parse).
    seq_to_acc = {sid: _accession_from_seq_id(sid) for sid in seq_ids}
    accs = sorted({a for a in seq_to_acc.values() if a})

    if not accs:
        # No UniProt-style IDs to look up — write an empty table so the
        # caller can distinguish "no taxonomy possible" from "not yet built".
        df = pd.DataFrame({
            "sequence_id": seq_ids,
            "uniprot_id": [seq_to_acc[s] for s in seq_ids],
            "organism": [None] * len(seq_ids),
            "domain": [None] * len(seq_ids),
            "phylum": [None] * len(seq_ids),
        })
        try:
            df.to_parquet(cache, index=False)
        except Exception:
            pass
        return df

    # Batch the API calls.
    rows: List[dict] = []
    for i in range(0, len(accs), _BATCH_SIZE):
        rows.extend(_fetch_batch(accs[i:i + _BATCH_SIZE]))
    by_acc = {r["uniprot_id"]: r for r in rows if r.get("uniprot_id")}

    # Build the per-sequence frame.
    records = []
    for sid in seq_ids:
        acc = seq_to_acc.get(sid)
        info = by_acc.get(acc) if acc else {}
        rec = {
            "sequence_id": sid,
            "uniprot_id": acc,
            "organism": info.get("organism"),
            "taxon_id": info.get("taxon_id"),
            "domain": info.get("domain"),
            "lineage": "; ".join(info.get("lineage") or []),
            "taxonomy_source": "uniprot" if info else None,
        }
        for rank in _RANK_LEVELS:
            rec[rank] = info.get(rank)
        records.append(rec)
    df = pd.DataFrame(records)

    # Annotate growth temps. The Enqvist dataset matches by organism
    # scientific name; rows with no organism will get NaN.
    try:
        from ..temperature import annotate_temperature
        df = annotate_temperature(df, organism_col="organism")
    except Exception:
        df["growth_temp"] = None

    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df
