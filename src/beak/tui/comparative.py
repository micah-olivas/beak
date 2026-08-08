"""Build per-target-position differential enrichment from a trait column.

Pipeline:
    1. Load the alignment for the active set.
    2. Load the joined `traits.parquet` for that set; pull the requested
       trait column and align it to the alignment record IDs.
    3. Coerce to numeric (booleans → 0/1; strings → NaN unless they
       parse). Skip rows with no value.
    4. Run `target_position_scores` with the supplied threshold; cache
       the resulting array as `comparative/<slug>.npy`.

The structure-view's "differential" color mode reads
`comparative/active.npy`, a symlink/copy pointing at the most recent
build. That keeps the consumer simple while still letting projects
hold multiple cached comparisons.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_") or "trait"


def _coerce_to_value(v) -> Optional[float]:
    """Best-effort numeric coercion for trait values.

    Booleans → 1.0 / 0.0. Floats / ints pass through. Strings that look
    like numbers parse; everything else (including ``"true"``/``"false"``
    strings that survived the loader's stringify pass) returns None so
    the caller drops the row.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return None if pd.isna(f) else f
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true"}:
            return 1.0
        if s in {"false"}:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _load_records(alignment_path: Path) -> List[Tuple[str, str]]:
    from ..alignments.cache import load_alignment_records
    # Names are already first-whitespace tokens — that's how Bio.SeqIO
    # populates `r.id`, and the cache preserves the same id field — so
    # downstream taxonomy / trait joins still match.
    return [(name, seq.upper()) for name, seq in load_alignment_records(alignment_path)]


def trait_columns(project) -> List[str]:
    """Names of available trait columns for the active set, ordered by coverage."""
    traits_path = project.active_homologs_dir() / "traits.parquet"
    if not traits_path.exists():
        return []
    try:
        df = pd.read_parquet(traits_path)
    except Exception:
        return []
    cols = [c for c in df.columns
            if c.startswith("trait_") and c != "trait_match_level"]
    cov = [(c, int(df[c].notna().sum())) for c in cols]
    cov.sort(key=lambda kv: kv[1], reverse=True)
    return [c for c, n in cov if n > 0]


def trait_summary(project, column: str) -> Optional[dict]:
    """Coverage / dtype / range hint for one trait column.

    Returned shape:
        {"n": int, "kind": "binary"|"numeric"|"categorical",
         "min": float|None, "max": float|None, "median": float|None,
         "values": [str, ...]}  # for categorical
    """
    traits_path = project.active_homologs_dir() / "traits.parquet"
    if not traits_path.exists():
        return None
    try:
        df = pd.read_parquet(traits_path)
    except Exception:
        return None
    if column not in df.columns:
        return None
    s = df[column].dropna()
    if len(s) == 0:
        return {"n": 0, "kind": "empty"}
    coerced = s.map(_coerce_to_value).dropna()
    n_numeric = len(coerced)
    if n_numeric > 0 and n_numeric / max(len(s), 1) > 0.5:
        unique = coerced.unique()
        if set(unique).issubset({0.0, 1.0}):
            return {
                "n": int(n_numeric), "kind": "binary",
                "min": 0.0, "max": 1.0, "median": float(coerced.median()),
            }
        return {
            "n": int(n_numeric), "kind": "numeric",
            "min": float(coerced.min()), "max": float(coerced.max()),
            "median": float(coerced.median()),
        }
    return {
        "n": int(len(s)), "kind": "categorical",
        "values": list(s.astype(str).value_counts().head(10).index),
    }


def build_comparative(
    project,
    column: str,
    threshold: float,
    set_active: bool = True,
) -> Optional[np.ndarray]:
    """Materialise per-target-position enrichment for one (trait, threshold).

    Returns the array (length = target sequence length) or None on
    failure. Cached at ``comparative/<slug>__<thr>.npy`` and (if
    ``set_active``) copied to ``comparative/active.npy`` so the
    structure-view picks it up on its next render.
    """
    homologs_dir = project.active_homologs_dir()
    aln_path = homologs_dir / "alignment.fasta"
    traits_path = homologs_dir / "traits.parquet"
    if not aln_path.exists() or not traits_path.exists():
        return None

    try:
        traits = pd.read_parquet(traits_path)
    except Exception:
        return None
    if column not in traits.columns:
        return None

    records = _load_records(aln_path)
    if len(records) < 3:
        return None

    # Align trait values to alignment record IDs.
    by_seq = dict(zip(traits["sequence_id"].astype(str), traits[column]))
    target_id = records[0][0]
    values: List[Optional[float]] = []
    for sid, _ in records:
        if sid == target_id:
            values.append(None)  # exclude target from its own comparison
            continue
        raw = by_seq.get(sid)
        values.append(_coerce_to_value(raw))

    from ..analysis.comparative import target_position_scores
    try:
        scores = target_position_scores(
            records, target_id, values, threshold=float(threshold)
        )
    except ValueError:
        # Threshold split degenerate (all-high or all-low). Caller can
        # surface this via trait_summary's median to suggest a better split.
        return None

    # Comparative scores are scoped to a single homolog set's
    # alignment, so every cache file lives under that set's directory.
    # The legacy project-level path (`project/comparative/`) is left
    # untouched for backward compat with old projects, but new writes
    # always land here.
    out_dir = project.homologs_set_dir(project.active_set_name()) / "comparative"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_slug(column)}__{threshold:g}.npy"
    np.save(out_dir / fname, scores)

    if set_active:
        np.save(out_dir / "active.npy", scores)
        # Record provenance in the manifest for the modal / display.
        with project.mutate() as m:
            comp = m.setdefault("comparative", {})
            comp["active_column"] = column
            comp["active_threshold"] = float(threshold)
            comp["active_n_high"] = int(sum(
                1 for v in values if v is not None and v >= float(threshold)
            ))
            comp["active_n_low"] = int(sum(
                1 for v in values if v is not None and v < float(threshold)
            ))
            comp["active_set"] = project.active_set_name()
            comp["last_updated"] = datetime.now()
    return scores


def load_active_scores(project) -> Optional[np.ndarray]:
    """Read the cached active comparative scores, length-checked against target.

    Returns None if (a) no cache exists for the active homolog set, or
    (b) the manifest's recorded `active_set` doesn't match the current
    one — in which case the cached scores belong to a different
    alignment and would mislead the structure ribbon.
    """
    active = project.active_set_name()
    # New per-set path first, then fall back to the legacy project-
    # level cache for projects created before per-set scoping landed.
    candidates = [
        project.homologs_set_dir(active) / "comparative" / "active.npy",
        project.path / "comparative" / "active.npy",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return None

    # Stale guard: refuse the legacy cache when the manifest says it
    # was computed for a different set.
    comp = project.manifest().get("comparative") or {}
    src_set = comp.get("active_set")
    if src_set and src_set != active:
        return None

    try:
        arr = np.load(p)
    except Exception:
        return None
    target = project.target_sequence() or ""
    if len(arr) != len(target):
        return None
    return arr


def differential_is_stale(project) -> bool:
    """True iff manifest's `comp.active_set` disagrees with the current set.

    The structure view uses this to render a stale-set badge so the
    user knows the cached differential coloring isn't from the current
    alignment.
    """
    comp = project.manifest().get("comparative") or {}
    src = comp.get("active_set")
    if not src:
        return False
    return src != project.active_set_name()


def active_label(project) -> Optional[str]:
    """Display label for the active comparative — e.g. `"temperature_growth ≥ 50"`."""
    m = project.manifest()
    comp = m.get("comparative") or {}
    col = comp.get("active_column")
    if not col:
        return None
    thr = comp.get("active_threshold")
    nice = col.removeprefix("trait_").replace("_", " ")
    if thr is None:
        return nice
    return f"{nice} >= {thr:g}"


# --------------------------------------------------------------------------- #
# Taxonomic clustering — group the MSA by lineage rank, score per-position
# clade bias. Sibling of build_comparative above: same cache-and-manifest
# shape, but the grouping variable is a taxonomy rank (N clades) rather
# than a thresholded trait (2 groups), and the controls exposed are the
# ones that keep the multi-group signal honest (weighting, permutation
# null, minimum clade size). Caches under the active set's `taxonomic/`
# dir and records provenance in manifest `["taxonomic"]` — a separate
# section from `["comparative"]` so the two coexist per set.
# --------------------------------------------------------------------------- #

# Coarse -> fine. `superkingdom` is the canonical top rank in
# taxonomy.parquet; `domain` is only a back-compat alias and is omitted
# here so it doesn't shadow superkingdom in the picker.
_RANK_ORDER = [
    "superkingdom", "phylum", "class", "order", "family", "genus", "species",
]


def rank_columns(project) -> List[str]:
    """Lineage-rank columns usable for clustering, coarse→fine.

    A rank qualifies if `taxonomy.parquet` has that column with at least
    two distinct non-null values (otherwise there's nothing to compare).
    Ordered by taxonomic depth so the picker reads superkingdom→species.
    """
    tax_path = project.active_homologs_dir() / "taxonomy.parquet"
    if not tax_path.exists():
        return []
    try:
        df = pd.read_parquet(tax_path)
    except Exception:
        return []
    out: List[str] = []
    for rank in _RANK_ORDER:
        if rank not in df.columns:
            continue
        vals = df[rank].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        if vals.nunique() >= 2:
            out.append(rank)
    return out


def rank_summary(project, rank: str, min_per_clade: int = 3) -> Optional[dict]:
    """Clade breakdown for one rank — coverage and how many clades qualify.

    Returned shape:
        {"n": annotated_sequences, "n_clades": distinct_clades,
         "n_qualifying": clades with >= min_per_clade sequences,
         "top": [(clade, size), ...]}  # largest first, up to 8
    """
    tax_path = project.active_homologs_dir() / "taxonomy.parquet"
    if not tax_path.exists():
        return None
    try:
        df = pd.read_parquet(tax_path)
    except Exception:
        return None
    if rank not in df.columns:
        return None
    s = df[rank].dropna()
    s = s[s.astype(str).str.strip() != ""].astype(str)
    if len(s) == 0:
        return {"n": 0, "n_clades": 0, "n_qualifying": 0, "top": []}
    counts = s.value_counts()
    qualifying = int((counts >= min_per_clade).sum())
    top = [(str(k), int(v)) for k, v in counts.head(8).items()]
    return {
        "n": int(len(s)),
        "n_clades": int(counts.size),
        "n_qualifying": qualifying,
        "top": top,
    }


def build_taxonomic(
    project,
    rank: str,
    *,
    min_per_clade: int = 3,
    use_weights: bool = True,
    n_permutations: int = 0,
    write_profiles: bool = False,
    set_active: bool = True,
) -> Optional[np.ndarray]:
    """Materialise per-target-position taxonomic bias for one rank.

    Groups the alignment by the chosen lineage `rank` (from
    `taxonomy.parquet`), scores each column's mutual information with clade
    membership via `analysis.comparative.target_taxonomic_scores`, and
    projects it onto target residue positions.

    Controls:
        min_per_clade: drop clades smaller than this before scoring.
        use_weights: Henikoff sequence weighting (defuse redundancy).
        n_permutations: label-shuffle null size; 0 → uncertainty
            coefficient, >0 → per-position z-score.
        write_profiles: also persist per-clade PSSM profiles (long-format
            `profiles__<rank>.parquet`: clade × alignment-column × AA ×
            freq) so a position's clade breakdown can be inspected, not
            just its aggregate score. This is the "output" control —
            score-only vs score + per-clade profiles.

    Returns the per-residue array (length = target sequence length) or
    None on failure (no alignment/taxonomy, degenerate split, etc.).
    Cached at `taxonomic/<rank>__<params>.npy` under the active set and,
    if `set_active`, copied to `taxonomic/active.npy` with provenance in
    manifest `["taxonomic"]`.
    """
    homologs_dir = project.active_homologs_dir()
    aln_path = homologs_dir / "alignment.fasta"
    tax_path = homologs_dir / "taxonomy.parquet"
    if not aln_path.exists() or not tax_path.exists():
        return None

    try:
        tax = pd.read_parquet(tax_path)
    except Exception:
        return None
    if rank not in tax.columns or "sequence_id" not in tax.columns:
        return None

    records = _load_records(aln_path)
    if len(records) < 3:
        return None

    # Map alignment record IDs to their clade label at this rank.
    by_seq = dict(zip(tax["sequence_id"].astype(str), tax[rank]))
    target_id = records[0][0]
    groups: List[Optional[str]] = []
    for sid, _ in records:
        if sid == target_id:
            groups.append(None)  # exclude target from its own comparison
            continue
        raw = by_seq.get(sid)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            groups.append(None)
            continue
        label = str(raw).strip()
        groups.append(label or None)

    from ..analysis.comparative import target_taxonomic_scores
    try:
        scores, result = target_taxonomic_scores(
            records,
            target_id,
            groups,
            use_weights=use_weights,
            min_per_clade=int(min_per_clade),
            n_permutations=int(n_permutations),
        )
    except ValueError:
        # Fewer than two clades cleared the size floor. Caller can surface
        # rank_summary's n_qualifying to suggest a coarser rank or lower
        # min_per_clade.
        return None

    out_dir = project.homologs_set_dir(project.active_set_name()) / "taxonomic"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{_slug(rank)}__mpc{int(min_per_clade)}"
        f"_w{int(bool(use_weights))}_p{int(n_permutations)}.npy"
    )
    np.save(out_dir / fname, scores)

    profiles_written = False
    if write_profiles:
        try:
            _write_clade_profiles(out_dir, records, groups, result.clades, rank)
            profiles_written = True
        except Exception:
            # Profiles are a convenience artefact; a failure here must not
            # sink the score build that already succeeded.
            profiles_written = False

    if set_active:
        np.save(out_dir / "active.npy", scores)
        with project.mutate() as m:
            tx = m.setdefault("taxonomic", {})
            tx["active_rank"] = rank
            tx["active_min_per_clade"] = int(min_per_clade)
            tx["active_use_weights"] = bool(use_weights)
            tx["active_n_permutations"] = int(n_permutations)
            tx["active_score_kind"] = result.score_kind
            tx["active_n_clades"] = len(result.clades)
            tx["active_clades"] = list(result.clades)
            tx["active_n_sequences"] = int(result.n_sequences)
            tx["active_has_profiles"] = profiles_written
            tx["active_set"] = project.active_set_name()
            tx["last_updated"] = datetime.now()
    return scores


def _write_clade_profiles(
    out_dir: Path,
    records: Sequence[Tuple[str, str]],
    groups: Sequence[Optional[str]],
    clades: Sequence[str],
    rank: str,
) -> None:
    """Persist per-clade PSSM profiles in long format for drill-down.

    Restricts to the clades that cleared the size floor (``clades``) so the
    profiles match the scored partition. Columns: ``clade``, ``column``
    (alignment column index), ``aa``, ``freq``.
    """
    from ..analysis.comparative import group_pssm

    keep = set(clades)
    prof_groups = [
        (g if (g is not None and str(g).strip() in keep) else None)
        for g in groups
    ]
    pssms = group_pssm(records, prof_groups)
    frames = []
    for clade, dfp in pssms.items():
        melted = (
            dfp.reset_index()
            .rename(columns={"index": "column"})
            .melt(id_vars="column", var_name="aa", value_name="freq")
        )
        melted["clade"] = clade
        frames.append(melted)
    if not frames:
        return
    prof = pd.concat(frames, ignore_index=True)[["clade", "column", "aa", "freq"]]
    prof.to_parquet(out_dir / f"profiles__{_slug(rank)}.parquet")


def load_active_taxonomic_scores(project) -> Optional[np.ndarray]:
    """Read cached active taxonomic scores, length-checked against target.

    Returns None if no cache exists for the active set, or the manifest's
    recorded `active_set` points at a different set (stale — the scores
    belong to another alignment's coordinate system).
    """
    active = project.active_set_name()
    p = project.homologs_set_dir(active) / "taxonomic" / "active.npy"
    if not p.exists():
        return None
    tx = project.manifest().get("taxonomic") or {}
    src_set = tx.get("active_set")
    if src_set and src_set != active:
        return None
    try:
        arr = np.load(p)
    except Exception:
        return None
    target = project.target_sequence() or ""
    if len(arr) != len(target):
        return None
    return arr


def taxonomic_is_stale(project) -> bool:
    """True iff manifest's `taxonomic.active_set` disagrees with the current set."""
    tx = project.manifest().get("taxonomic") or {}
    src = tx.get("active_set")
    if not src:
        return False
    return src != project.active_set_name()


def active_taxonomic_label(project) -> Optional[str]:
    """Display label for the active taxonomic clustering — e.g. `"phylum · 6 clades · z"`."""
    tx = project.manifest().get("taxonomic") or {}
    rank = tx.get("active_rank")
    if not rank:
        return None
    n = tx.get("active_n_clades")
    kind = tx.get("active_score_kind")
    suffix = "z" if kind == "permutation_zscore" else "U"
    if n:
        return f"{rank} · {n} clades · {suffix}"
    return rank
