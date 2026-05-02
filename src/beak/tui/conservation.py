"""Per-residue conservation, with an MSA fast-path.

If `homologs/alignment.fasta` is on disk, we use it directly: at each
target position (non-gap in the first sequence) we count the fraction
of homolog residues that match the target. This is the canonical way
to derive conservation once the alignment exists.

Before the alignment lands, we fall back to a pairwise version: for
each hit in `homologs/sequences.fasta`, do a fast local alignment to
the target and tally matches. Cheaper but rougher.

Either way the result is in [0, 100] so the existing pLDDT color
palette renders directly. Cached on disk under
`homologs/conservation.npy`; the cache is invalidated by the layers
panel after the alignment is pulled.
"""

from pathlib import Path
from typing import Optional

import numpy as np


_CACHE_FILENAME = "conservation.npy"
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
    from Bio import SeqIO

    records = list(SeqIO.parse(str(alignment_path), "fasta"))
    if len(records) < 2:
        return None

    aligned = [str(r.seq).upper() for r in records]
    target_aln = aligned[0]

    target_len = len(target_seq)
    counts = np.zeros(target_len, dtype=np.int32)
    totals = np.zeros(target_len, dtype=np.int32)

    target_pos = 0
    aln_len = len(target_aln)
    for col in range(aln_len):
        target_aa = target_aln[col]
        if target_aa == "-" or target_aa == ".":
            continue
        if target_pos >= target_len:
            break
        for seq in aligned:
            ha = seq[col] if col < len(seq) else "-"
            if ha != "-" and ha != ".":
                totals[target_pos] += 1
                if ha == target_aa:
                    counts[target_pos] += 1
        target_pos += 1

    if totals.sum() == 0:
        return None
    return np.where(totals > 0, counts / totals, 0.0).astype(np.float32) * 100.0


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
