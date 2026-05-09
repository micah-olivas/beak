"""Memmap-friendly cache for parsed FASTA alignments.

Re-parsing 20k+-sequence alignments through `Bio.SeqIO.parse` on every
project view is the dominant load cost in the TUI — typically ~5–10 s
per open. The first time we read a FASTA, we convert it to a uint8
matrix (one row per sequence, ASCII codes per column) plus a parallel
names array, and write both alongside the source file as
`<fasta>.npz`. Subsequent loads pull the arrays back in tens of
milliseconds without touching SeqIO.

The cache is keyed on the FASTA's mtime: any rewrite (e.g. a length
filter via `BeakProject.filter_homolog_set_by_length`) makes the
sidecar stale, and the next call rebuilds it transparently.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


_CACHE_EXT = ".npz"


def load_alignment_records(fasta_path: Path) -> List[Tuple[str, str]]:
    """Return `[(name, seq), ...]` parsed from a FASTA, using a sidecar
    `.npz` cache when fresh.

    Returns an empty list when the FASTA is missing or unparseable.
    Drop-in replacement for the common pattern
    ``[(r.id, str(r.seq)) for r in SeqIO.parse(str(path), "fasta")]``.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        return []

    cache_path = _cache_path_for(fasta_path)
    if _cache_fresh(cache_path, fasta_path):
        try:
            return _load_from_cache(cache_path)
        except Exception:
            # Fall through to a fresh parse on cache corruption — the
            # rebuild below will overwrite the bad sidecar.
            pass

    records = _parse_fasta(fasta_path)
    if records:
        try:
            _save_cache(cache_path, records, fasta_path)
        except Exception:
            # A failed cache write isn't fatal; the records are still
            # returned. Next call will retry the write.
            pass
    return records


def invalidate_cache(fasta_path: Path) -> None:
    """Remove the sidecar cache for a FASTA, if any. Cheap; no-op if absent."""
    cache_path = _cache_path_for(Path(fasta_path))
    try:
        cache_path.unlink(missing_ok=True)
    except OSError:
        pass


def _cache_path_for(fasta_path: Path) -> Path:
    # `with_name(...)` keeps both extensions visible (alignment.fasta.npz)
    # so the relationship to the source is obvious in the directory.
    return fasta_path.with_name(fasta_path.name + _CACHE_EXT)


def _cache_fresh(cache_path: Path, fasta_path: Path) -> bool:
    """Cache is fresh iff (mtime ≥ source mtime) AND (source size matches
    what was recorded at write-time).

    mtime alone is unsafe: a fast rewrite — common when a script
    regenerates the FASTA in the same wall-second, or when filesystems
    with second-resolution mtimes (NFS, FAT) collapse two writes into
    a single timestamp — can leave the sidecar pointing at stale
    bytes. The size cross-check makes that case detectable: if the
    new FASTA has a different on-disk size than what we cached, we
    know the sidecar is stale even if the mtimes agree.
    """
    if not cache_path.exists():
        return False
    try:
        cache_st = cache_path.stat()
        fasta_st = fasta_path.stat()
        if cache_st.st_mtime < fasta_st.st_mtime:
            return False
    except OSError:
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_size = int(data["src_size"].item()) if "src_size" in data.files else -1
    except Exception:
        return False
    # Pre-`src_size` sidecars store -1 → fall back to mtime-only
    # behavior. Subsequent writes upgrade them automatically.
    if cached_size < 0:
        return True
    return cached_size == fasta_st.st_size


def _load_from_cache(cache_path: Path) -> List[Tuple[str, str]]:
    # `allow_pickle=False` keeps the load fast and rules out arbitrary
    # code execution from a tampered sidecar — names are fixed-width
    # unicode (no objects), seqs are uint8.
    with np.load(cache_path, allow_pickle=False) as data:
        matrix = data["seqs"]   # (N, L) uint8
        names = data["names"]   # (N,) <U...

    if matrix.ndim != 2 or len(names) != matrix.shape[0]:
        raise ValueError("malformed alignment cache")

    return [
        (
            str(names[i]),
            # Sequences are right-padded with NUL bytes to the longest
            # row — strip those and decode the ASCII payload back to a
            # str so callers see the same shape SeqIO would have given.
            matrix[i].tobytes().rstrip(b"\x00").decode("ascii"),
        )
        for i in range(len(names))
    ]


def _save_cache(cache_path: Path, records: List[Tuple[str, str]], fasta_path: Path) -> None:
    n = len(records)
    width = max(len(seq) for _, seq in records)
    matrix = np.zeros((n, width), dtype=np.uint8)
    for i, (_, seq) in enumerate(records):
        b = seq.encode("ascii", errors="replace")
        matrix[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    names = np.array([r[0] for r in records])  # auto fixed-width unicode

    try:
        src_size = np.array(int(fasta_path.stat().st_size), dtype=np.int64)
    except OSError:
        src_size = np.array(-1, dtype=np.int64)

    # Atomic write: a partial sidecar would falsely look "fresh" by
    # mtime on the next open and break `_load_from_cache`.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    np.savez(tmp, seqs=matrix, names=names, src_size=src_size)
    tmp.replace(cache_path)


def _parse_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    from Bio import SeqIO
    return [
        (r.id, str(r.seq))
        for r in SeqIO.parse(str(fasta_path), "fasta")
    ]
