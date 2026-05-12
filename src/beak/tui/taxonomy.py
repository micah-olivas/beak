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
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd


def atomic_to_parquet(df: pd.DataFrame, path, **kwargs) -> None:
    """Write a DataFrame to a parquet file via temp-and-replace.

    Without this, a SIGINT or process kill in the middle of
    ``df.to_parquet(path, …)`` leaves a truncated file at ``path``
    that the next read raises on (`pyarrow.lib.ArrowInvalid: Couldn't
    deserialise Thrift`). The two-step write-then-rename ensures
    `path` either points at the previous good copy or the fresh one
    — never a torn write.

    Cleans up the temp file on any error so a failed write doesn't
    leave a `.tmp.parquet` orphan in the project directory. Used by
    every parquet writer in `tui/`.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        # If `to_parquet` raised, `os.replace` was never reached and
        # `tmp` is still on disk. Same goes for an exception in the
        # caller's `try` that wraps this. Best-effort cleanup either
        # way; ignore if the file was already moved.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass

# Signature of the progress callback `build_taxonomy_table` invokes
# after each UniProt batch. Receives (sequences_done, sequences_total);
# called from the worker thread, so callers must marshal back to the UI
# thread themselves (e.g. `app.call_from_thread`).
ProgressCb = Optional[Callable[[int, int], None]]


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


def _maybe_merge_mmseqs_lca(
    df: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """Fold `taxonomy_mmseqs.parquet` (LCA fallback) into the main frame.

    The MMseqs2 LCA fallback writes its results to a sibling parquet
    (`taxonomy_mmseqs.parquet`) rather than merging directly into the
    main `taxonomy.parquet`. So sequences that UniProt couldn't resolve
    but LCA could (typically the +500–1000 gain in `n_assigned`) stay
    invisible to the taxonomy view, which only reads `taxonomy.parquet`.
    This merges them in: for any row in the main frame with a missing
    domain whose accession also appears in the LCA results, the LCA's
    rank columns (domain / phylum / class / order / family / genus /
    species + scientific name + full lineage) are copied across, and
    `taxonomy_source` is set to ``"mmseqs_lca"`` so callers can
    distinguish UniProt-authoritative rows from inferred ones.

    No-op when the LCA sidecar doesn't exist or doesn't share any
    accessions with the main frame's unresolved rows.
    """
    mmseqs_path = cache_path.with_name("taxonomy_mmseqs.parquet")
    if not mmseqs_path.exists():
        return df
    if "sequence_id" not in df.columns:
        return df

    try:
        mmseqs = pd.read_parquet(mmseqs_path)
    except Exception:
        return df
    if mmseqs.empty or "query" not in mmseqs.columns:
        return df

    # Build accession→main_row_index map for rows that the UniProt
    # path didn't resolve (where `domain` is null). LCA only has
    # something to add for those.
    if "domain" not in df.columns:
        return df
    unresolved_mask = df["domain"].isna()
    if not unresolved_mask.any():
        return df

    acc_to_idx: Dict[str, int] = {}
    for idx, sid in df.loc[unresolved_mask, "sequence_id"].items():
        acc = _accession_from_seq_id(str(sid)) if sid else None
        if acc:
            acc_to_idx[acc] = idx

    # LCA's `query` column carries the bare accession.
    mmseqs_indexed = (
        mmseqs.dropna(subset=["query"])
        .drop_duplicates(subset=["query"], keep="first")
        .set_index("query")
    )

    rank_cols = (
        "domain", "kingdom", "phylum", "class",
        "order", "family", "genus", "species",
    )
    n_filled = 0
    out = df.copy()
    if "taxonomy_source" not in out.columns:
        out["taxonomy_source"] = pd.NA
    if "lineage" not in out.columns:
        out["lineage"] = pd.NA
    if "organism" not in out.columns:
        out["organism"] = pd.NA

    for acc, main_idx in acc_to_idx.items():
        if acc not in mmseqs_indexed.index:
            continue
        mrow = mmseqs_indexed.loc[acc]
        for col in rank_cols:
            if col in mmseqs_indexed.columns:
                v = mrow[col]
                if pd.notna(v):
                    out.at[main_idx, col] = v
        if "scientific_name" in mmseqs_indexed.columns:
            sn = mrow["scientific_name"]
            if pd.notna(sn):
                out.at[main_idx, "organism"] = sn
        if "lineage" in mmseqs_indexed.columns:
            ln = mrow["lineage"]
            if pd.notna(ln):
                out.at[main_idx, "lineage"] = ln
        out.at[main_idx, "taxonomy_source"] = "mmseqs_lca"
        n_filled += 1

    if n_filled == 0:
        return df

    # Re-run the Enqvist join on the merged frame so LCA-derived
    # organism names also get a `growth_temp` lookup. Done here
    # rather than relying on `_maybe_reanneal_growth_temp` because
    # that helper's early-return fires as soon as *any* row has a
    # non-NaN growth_temp — leaving the new LCA rows uncovered.
    try:
        from ..temperature import annotate_temperature
        cleaned = out.drop(
            columns=["growth_temp", "temp_source"], errors="ignore"
        )
        out = annotate_temperature(cleaned, organism_col="organism")
    except Exception:
        pass

    try:
        atomic_to_parquet(out, cache_path, index=False)
    except Exception:
        pass
    return out


def _maybe_reanneal_growth_temp(
    df: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """Recompute `growth_temp` on a cached frame if it's all-NaN/missing.

    Older parquets were written before `annotate_temperature` could
    parse "Escherichia coli" against the Enqvist dataset's
    "escherichia_coli". They land here with an empty growth_temp
    column even though the organism column is populated — recompute
    the temperature locally and overwrite the cache so the next read
    is fast again.
    """
    has_growth = "growth_temp" in df.columns
    has_some = has_growth and df["growth_temp"].notna().any()
    if has_some:
        return df  # already populated — nothing to do
    if "organism" not in df.columns or df["organism"].dropna().empty:
        return df  # no organisms to look up either; bail

    try:
        from ..temperature import annotate_temperature
        # Drop the empty growth_temp / temp_source columns first so the
        # merge doesn't append "_x"/"_y" suffixes alongside the originals.
        cleaned = df.drop(
            columns=["growth_temp", "temp_source"], errors="ignore"
        )
        rebuilt = annotate_temperature(cleaned, organism_col="organism")
    except Exception:
        return df  # silent fallback — never let a bad parquet block the view

    try:
        atomic_to_parquet(rebuilt, cache_path, index=False)
    except Exception:
        pass
    return rebuilt


def build_taxonomy_table(
    project,
    force: bool = False,
    progress_cb: ProgressCb = None,
) -> Optional[pd.DataFrame]:
    """Materialise `homologs/taxonomy.parquet` for a project.

    Returns the DataFrame, or None if no homologs exist. Cached: returns
    the existing file if present unless `force=True`. When `progress_cb`
    is given, it's called as `(done, total)` after each UniProt batch
    so the UI can render a live progress indicator — at ~50 accessions
    per round-trip and ~1 s per call, a 25k-sequence build runs for
    several minutes and benefits from incremental feedback.
    """
    homologs_dir = project.active_homologs_dir()
    fasta = homologs_dir / "sequences.fasta"
    if not fasta.exists():
        return None

    cache = homologs_dir / "taxonomy.parquet"
    cached: Optional[pd.DataFrame] = None
    if cache.exists() and not force:
        try:
            cached = pd.read_parquet(cache)
        except Exception:
            cached = None  # bad cache; fall through to rebuild
        else:
            # One-shot migration steps for cached parquets:
            #   1. Fold in the MMseqs2 LCA fallback's results, which
            #      previously sat orphaned in `taxonomy_mmseqs.parquet`
            #      and were never visible to the taxonomy view —
            #      explaining why the unresolved bar stayed at 56%
            #      even after the fallback completed.
            #   2. Re-anneal `growth_temp` using the fixed normalization
            #      (handles parquets written before the Enqvist
            #      organism-name fix landed). Run this *after* the
            #      merge so any organism names the LCA contributed
            #      also get a temperature lookup.
            cached = _maybe_merge_mmseqs_lca(cached, cache)
            cached = _maybe_reanneal_growth_temp(cached, cache)

    seq_ids = _parse_fasta_ids(fasta)
    if not seq_ids:
        return cached if cached is not None else None

    # Coverage check: if the cached parquet predates a homologs growth
    # (e.g. user re-ran search and the FASTA went from 80 → 378), its
    # `sequence_id` set won't cover the FASTA. Without this check the
    # taxonomy view would silently keep showing the old subset (which
    # was the symptom: project had 378 hits but taxonomy view said 80).
    cached_seq_ids: set = set()
    if cached is not None and "sequence_id" in cached.columns:
        cached_seq_ids = set(cached["sequence_id"].astype(str))
    missing_seq_ids = [s for s in seq_ids if s not in cached_seq_ids]

    if not missing_seq_ids and cached is not None and not force:
        return cached

    if progress_cb is not None:
        try:
            progress_cb(0, len(missing_seq_ids) or len(seq_ids))
        except Exception:
            pass

    # Build accession map for the rows we still need to fetch. On a
    # full rebuild (no cache, or `force=True`) this is every seq_id;
    # on an incremental top-up it's just the new ones. Either way the
    # batch / progress code below operates on `target_seq_ids`.
    target_seq_ids = missing_seq_ids if (cached is not None and not force) else seq_ids
    seq_to_acc = {sid: _accession_from_seq_id(sid) for sid in target_seq_ids}
    accs = sorted({a for a in seq_to_acc.values() if a})

    if not accs:
        # No UniProt-style IDs to look up. Build placeholder rows for
        # the missing seq_ids and merge them in (so an unresolvable
        # FASTA still produces a complete-coverage parquet).
        empty_records = [
            {
                "sequence_id": sid,
                "uniprot_id": seq_to_acc.get(sid),
                "organism": None,
                "domain": None,
                "phylum": None,
            }
            for sid in target_seq_ids
        ]
        new_df = pd.DataFrame(empty_records)
        df = (
            pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates(subset=["sequence_id"], keep="first")
            if cached is not None and not cached.empty else new_df
        )
        try:
            atomic_to_parquet(df, cache, index=False)
        except Exception:
            pass
        return df

    # Batch the API calls. Progress is reported in *sequence* units
    # (not accession units) so the UI counter matches the user's mental
    # model — a 25k-hit set polls 25k regardless of how many duplicate
    # accessions get folded into one UniProt batch.
    rows: List[dict] = []
    n_total = len(target_seq_ids)
    n_accs = len(accs)
    for i in range(0, n_accs, _BATCH_SIZE):
        rows.extend(_fetch_batch(accs[i:i + _BATCH_SIZE]))
        if progress_cb is not None:
            done = int(min(n_accs, i + _BATCH_SIZE) / n_accs * n_total) \
                if n_accs else n_total
            try:
                progress_cb(done, n_total)
            except Exception:
                pass
    by_acc = {r["uniprot_id"]: r for r in rows if r.get("uniprot_id")}

    # Build per-sequence rows for the target seq_ids only.
    records = []
    for sid in target_seq_ids:
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
    new_df = pd.DataFrame(records)

    # Annotate growth temps. The Enqvist dataset matches by organism
    # scientific name; rows with no organism will get NaN.
    try:
        from ..temperature import annotate_temperature
        new_df = annotate_temperature(new_df, organism_col="organism")
    except Exception:
        new_df["growth_temp"] = None

    # Merge cached + newly-fetched rows. `keep="first"` preserves the
    # cached row when a `sequence_id` somehow exists in both (e.g. an
    # MMseqs2 LCA fallback already populated organism for that row —
    # we don't want to clobber it with a possibly-empty UniProt result).
    if cached is not None and not cached.empty:
        df = (
            pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates(subset=["sequence_id"], keep="first")
                .reset_index(drop=True)
        )
    else:
        df = new_df

    try:
        atomic_to_parquet(df, cache, index=False)
    except Exception:
        pass
    return df


# --- MMseqs2 LCA fallback ---------------------------------------------------
#
# UniProt's REST endpoint resolves classic accessions (`P00533`, `Q9Y6Q9`)
# but not UniParc-style IDs (`UPI002E2621C6`) or non-UniProt headers from
# BFD / metagenomic searches. Those rows land in `taxonomy.parquet` with
# `organism = None` and `taxonomy_source = None`. The fallback below runs
# MMseqs2 LCA only on those unresolved sequences (small server cost) and
# merges the results back into the canonical table.


def unresolved_seq_ids(project, set_name: Optional[str] = None) -> List[str]:
    """Sequence IDs in a set's taxonomy.parquet with no organism.

    Uses ``set_name`` when provided; falls back to the active set.
    Returns an empty list when the parquet doesn't exist yet — callers
    should treat that as "build the canonical table first."
    """
    if set_name is not None:
        cache = project.homologs_set_dir(set_name) / "taxonomy.parquet"
    else:
        cache = project.active_homologs_dir() / "taxonomy.parquet"
    if not cache.exists():
        return []
    try:
        df = pd.read_parquet(cache)
    except Exception:
        return []
    if "sequence_id" not in df.columns or "organism" not in df.columns:
        return []
    missing = df[df["organism"].isna()]
    return [str(s) for s in missing["sequence_id"].tolist() if s]


def write_fallback_fasta(
    project, seq_ids: List[str], set_name: Optional[str] = None
) -> Optional[Path]:
    """Stage a small FASTA with just the unresolved sequences.

    Reads ``homologs/sets/<set>/sequences.fasta`` line-by-line and writes
    only records whose header token matches a target ID. Uses ``set_name``
    when provided; falls back to the active set. Returns the path to the
    staged file, or None if no matching records were found. Stable
    filename so re-runs overwrite cleanly.
    """
    if set_name is not None:
        homologs_dir = project.homologs_set_dir(set_name)
    else:
        homologs_dir = project.active_homologs_dir()
    src = homologs_dir / "sequences.fasta"
    if not src.exists() or not seq_ids:
        return None

    target = set(seq_ids)
    out = homologs_dir / "taxonomy_fallback.fasta"
    keep = False
    n_written = 0
    with open(src) as f, open(out, "w") as g:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip().split()[0] if len(line) > 1 else ""
                keep = header in target
                if keep:
                    g.write(line)
                    n_written += 1
                continue
            if keep:
                g.write(line)
    if n_written == 0:
        try:
            out.unlink()
        except OSError:
            pass
        return None
    return out


def merge_mmseqs_fallback(project) -> int:
    """Upsert MMseqs LCA assignments into the canonical taxonomy.parquet.

    Reads `taxonomy_mmseqs.parquet` (written by `_pull_taxonomy_now`)
    and fills any rows in `taxonomy.parquet` that still have a null
    organism. Doesn't touch rows that already have a UniProt-resolved
    annotation — UniProt is more authoritative for clade depth.

    Returns the number of rows that gained an organism. Re-runs are
    idempotent: rows that were already filled stay put.
    """
    homologs_dir = project.active_homologs_dir()
    canonical = homologs_dir / "taxonomy.parquet"
    fallback = homologs_dir / "taxonomy_mmseqs.parquet"
    if not canonical.exists() or not fallback.exists():
        return 0
    try:
        canon = pd.read_parquet(canonical)
        fb = pd.read_parquet(fallback)
    except Exception:
        return 0
    if "sequence_id" not in canon.columns or "organism" not in canon.columns:
        return 0

    # MMseqs LCA output uses different column names depending on the
    # parser path. Normalize to the canonical schema's column names.
    rename = {
        "query": "sequence_id",
        "scientific_name": "organism",
        "lineage_string": "lineage",
    }
    fb = fb.rename(columns={k: v for k, v in rename.items() if k in fb.columns})
    if "sequence_id" not in fb.columns:
        return 0

    fb_by_id = {
        str(row["sequence_id"]): row for _, row in fb.iterrows()
    }

    # Columns we may upsert from the LCA frame. Only fill where
    # canonical was null — and we tag the source so downstream
    # analysis can weight UniProt-resolved rows differently.
    fillable = [
        "organism", "taxon_id", "domain", "lineage",
        "superkingdom", "kingdom", "phylum", "class",
        "order", "family", "genus", "species",
    ]

    n_filled = 0
    for i, row in canon.iterrows():
        if pd.notna(row.get("organism")):
            continue
        sid = str(row.get("sequence_id") or "")
        fb_row = fb_by_id.get(sid)
        if fb_row is None:
            continue
        gained = False
        for col in fillable:
            if col not in fb.columns:
                continue
            val = fb_row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if col not in canon.columns:
                canon[col] = None
            canon.at[i, col] = val
            if col == "organism":
                gained = True
        if gained:
            canon.at[i, "taxonomy_source"] = "mmseqs_lca"
            n_filled += 1

    if n_filled == 0:
        return 0

    # Re-anneal growth_temp on the freshly filled rows so the
    # downstream views (PCA temp coloring, taxonomy histogram) pick
    # them up without a separate manual rebuild.
    try:
        from ..temperature import annotate_temperature
        # Rebuild only the temp columns; preserve all others.
        without_temps = canon.drop(
            columns=["growth_temp", "temp_source"], errors="ignore"
        )
        canon = annotate_temperature(without_temps, organism_col="organism")
    except Exception:
        pass

    try:
        atomic_to_parquet(canon, canonical, index=False)
    except Exception:
        pass
    return n_filled
