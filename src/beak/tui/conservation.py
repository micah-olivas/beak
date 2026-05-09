"""Per-residue conservation, with an MSA fast-path.

If ``homologs/alignment.fasta`` is on disk, we score every column with
**Jensen-Shannon divergence against a BLOSUM62 background distribution**
(Capra & Singh, *Bioinformatics* 2007 — the field-standard metric per
several head-to-head comparison studies, see
``alignments/conservation.py``) and project the per-column values onto
target positions.

Before the alignment lands, we fall back to a quick-and-dirty pairwise
identity proxy: for each hit in ``homologs/sequences.fasta``, do a
local alignment to the target and tally per-position match fractions.
Cheaper but rougher; gets replaced once the MSA pull completes and the
JSD path takes over.

Either way the result is in [0, 100] so the existing pLDDT-style colour
palette renders directly. Cached on disk under
``homologs/conservation_jsd.npy``; the cache is dropped by the layers
panel after a new alignment is pulled. (The pre-2026 cache filename
``conservation.npy`` is silently swept on first read so projects that
predate the JSD switch don't keep serving stale target-identity scores.)
"""

from pathlib import Path
from typing import Optional

import numpy as np


_CACHE_FILENAME = "conservation_jsd.npy"
# Older cache file from beak's pre-2026 target-identity era. Cleaned up
# on read so existing projects don't keep silently using the old
# metric forever.
_LEGACY_CACHE_FILENAME = "conservation.npy"
_MAX_HITS = 500  # cap pairwise work to keep first-render under a couple seconds


def compute_quick_conservation(project) -> Optional[np.ndarray]:
    """Return a per-residue conservation array, or None if no homologs.

    Prefers the MSA at `homologs/alignment.fasta` when available;
    otherwise falls back to pairwise alignments of `sequences.fasta`.
    Cached on disk; delete the cache file to recompute.
    """
    homologs_dir = project.active_homologs_dir()
    target_seq = project.target_sequence()
    if not target_seq:
        return None

    # Sweep stale legacy caches once. Pre-2026 beak wrote per-position
    # target-identity scores to `conservation.npy`; the new file uses
    # JSD against a BLOSUM62 background and lives at a different name
    # so the old values can't be mistaken for the new ones. The
    # legacy file is harmless to leave on disk, but removing it keeps
    # `du` honest and avoids confusion when someone greps for it.
    legacy_cache = homologs_dir / _LEGACY_CACHE_FILENAME
    if legacy_cache.exists():
        try:
            legacy_cache.unlink()
        except OSError:
            pass

    cache_path = homologs_dir / _CACHE_FILENAME
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            if len(cached) == len(target_seq):
                return cached
        except Exception:
            pass

    alignment_path = homologs_dir / "alignment.fasta"
    if alignment_path.exists():
        result = _conservation_from_msa(alignment_path, target_seq)
        if result is not None:
            _save_cache(cache_path, result)
            return result

    hits_fasta = homologs_dir / "sequences.fasta"
    if hits_fasta.exists():
        result = _conservation_pairwise(hits_fasta, target_seq)
        if result is not None:
            _save_cache(cache_path, result)
            return result

    return None


def _conservation_from_msa(alignment_path: Path, target_seq: str) -> Optional[np.ndarray]:
    from ..alignments.cache import load_alignment_records
    from ..alignments.conservation import (
        conservation_score, project_to_target,
    )

    records = load_alignment_records(alignment_path)
    if len(records) < 2:
        return None

    aligned = [seq.upper() for _, seq in records]
    # Identify the target row by ungapped-sequence equality with the
    # passed-in `target_seq` rather than assuming `records[0]`. An MSA
    # written by a third-party tool or sorted by id may not have the
    # target first — using the wrong row would compute conservation
    # against a homolog and silently mis-color the structure view.
    target_seq_upper = target_seq.upper()
    target_aln: Optional[str] = None
    for seq in aligned:
        if seq.replace("-", "").replace(".", "") == target_seq_upper:
            target_aln = seq
            break
    if target_aln is None:
        # Fallback to the legacy first-row assumption (correct for MSAs
        # beak's own pipeline produces) — better than refusing to compute
        # when an external tool reordered the file but kept the target
        # in row 0 anyway.
        target_aln = aligned[0]

    # Field-standard Jensen-Shannon divergence against a BLOSUM62
    # background (Capra & Singh 2007). Replaces the prior per-position
    # target-identity metric, which weighed gaps and common residues
    # equally with rare ones and made functionally important sites less
    # distinguishable. Score is in [0, 1] from the new module; we scale
    # to [0, 100] to keep the existing pLDDT-band palette and persisted
    # midpoint values working unchanged.
    column_scores = conservation_score(
        aligned,
        method="js_divergence",
        background="blosum62",
        gap_penalty=True,
    )
    if column_scores.size == 0:
        return None

    per_target = project_to_target(column_scores, target_aln, len(target_seq))
    return (per_target * 100.0).astype(np.float32)


def _conservation_pairwise(hits_fasta: Path, target_seq: str) -> Optional[np.ndarray]:
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    n = len(target_seq)
    counts = np.zeros(n, dtype=np.int32)
    totals = np.zeros(n, dtype=np.int32)

    n_hits = 0
    for record in SeqIO.parse(str(hits_fasta), "fasta"):
        if n_hits >= _MAX_HITS:
            break
        hit_seq = str(record.seq).upper().replace("*", "")
        if not hit_seq:
            continue
        try:
            best = aligner.align(target_seq, hit_seq)[0]
        except Exception:
            continue

        target_blocks, hit_blocks = best.aligned
        for (t_start, t_end), (h_start, h_end) in zip(target_blocks, hit_blocks):
            for tp, hp in zip(range(t_start, t_end), range(h_start, h_end)):
                if tp < n:
                    totals[tp] += 1
                    if target_seq[tp] == hit_seq[hp]:
                        counts[tp] += 1
        n_hits += 1

    if n_hits == 0:
        return None
    return np.where(totals > 0, counts / totals, 0.0).astype(np.float32) * 100.0


def _save_cache(cache_path: Path, arr: np.ndarray) -> None:
    try:
        np.save(cache_path, arr)
    except Exception:
        pass
