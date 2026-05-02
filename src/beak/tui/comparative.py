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
    from Bio import SeqIO
    records = []
    for r in SeqIO.parse(str(alignment_path), "fasta"):
        # Match the FASTA id format the taxonomy/traits builders use
        # (first whitespace-delimited token from the header).
        records.append((r.id, str(r.seq).upper()))
    return records


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
