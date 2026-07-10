"""Local structural homology search via foldseek.

This is beak's first genuinely local heavy-compute path. Unlike sequence
search (MMseqs2 run remotely over SSH via ``RemoteJobManager``), foldseek
runs as a synchronous local subprocess against a locally-downloaded
structure database — closer in spirit to ``extract_structure_features``
than to the remote job managers.

The binary is resolved from config (``foldseek.binary``) with a
``shutil.which`` fallback, mirroring the mkdssp discovery in
``structures/features.py``: check, run, error/degrade cleanly.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


# Prebuilt databases foldseek can download via `foldseek databases`. Not
# exhaustive — foldseek accepts any name it knows — but the common ones,
# used to populate CLI help and to sanity-check `beak foldseek setup`.
KNOWN_DATABASES = (
    "PDB",
    "Alphafold/UniProt",
    "Alphafold/UniProt50",
    "Alphafold/Proteome",
    "Alphafold/Swiss-Prot",
    "ESMAtlas30",
    "CATH50",
    "BFVD",
)

# Columns always requested from foldseek. Beyond the default m8 set we add
# alntmscore (global structural similarity, 0-1), lddt (local accuracy,
# 0-1), and query/target coverage — the fields worth landing on a
# structural-neighbours table. Order here IS the output column order.
DEFAULT_OUTPUT_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "lddt", "qcov", "tcov",
]

# Column typing applied after the raw tab-split parse.
_FLOAT_COLS = {"fident", "evalue", "bits", "alntmscore", "lddt", "qcov", "tcov"}
_INT_COLS = {"alnlen", "mismatch", "gapopen", "qstart", "qend", "tstart", "tend"}


class FoldseekError(RuntimeError):
    """A foldseek invocation failed, or the binary/database is missing."""


def resolve_foldseek_binary(explicit: Optional[str] = None) -> str:
    """Locate the foldseek executable.

    Resolution order: an ``explicit`` path/name if given, then the
    ``foldseek.binary`` config key, then ``foldseek`` on ``$PATH``.
    Raises :class:`FoldseekError` with an actionable message if none
    resolve — the same check-then-error idiom as ``_dssp_available``.
    """
    def _resolve(cand: str) -> Optional[str]:
        found = shutil.which(cand)
        if found:
            return found
        p = Path(cand).expanduser()
        return str(p) if p.exists() else None

    if explicit:
        resolved = _resolve(explicit)
        if resolved:
            return resolved
        raise FoldseekError(f"foldseek binary not found: {explicit}")

    from ..config import get_foldseek_config
    configured = get_foldseek_config().get("binary")
    if configured:
        resolved = _resolve(configured)
        if resolved:
            return resolved

    found = shutil.which("foldseek")
    if found:
        return found

    raise FoldseekError(
        "foldseek not found on PATH. Install it "
        "(conda install -c conda-forge -c bioconda foldseek) or set the path "
        "with `beak config set foldseek.binary /path/to/foldseek`."
    )


def foldseek_version(binary: Optional[str] = None) -> Optional[str]:
    """Return the foldseek version string, or None if unavailable."""
    try:
        resolved = resolve_foldseek_binary(binary)
    except FoldseekError:
        return None
    try:
        proc = subprocess.run([resolved, "version"],
                              capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode == 0:
        return proc.stdout.strip() or None
    return None


def parse_foldseek_m8(text: str,
                      columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Parse foldseek tab-separated m8 output into a typed DataFrame.

    Pure function — no subprocess — so it is unit-testable against
    captured output. ``columns`` must match the ``--format-output`` used
    to produce ``text`` (defaults to :data:`DEFAULT_OUTPUT_COLUMNS`).
    Empty input yields an empty DataFrame with the right columns. Rows
    with too few/many fields are padded/truncated rather than dropped, so
    a stray column never desyncs the whole table.
    """
    cols = list(columns or DEFAULT_OUTPUT_COLUMNS)
    rows: List[List[Optional[str]]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields: List[Optional[str]] = line.split("\t")
        if len(fields) < len(cols):
            fields = fields + [None] * (len(cols) - len(fields))
        elif len(fields) > len(cols):
            fields = fields[:len(cols)]
        rows.append(fields)

    df = pd.DataFrame(rows, columns=cols)
    for col in cols:
        if col in _FLOAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col in _INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def run_easy_search(
    query_cif: str,
    db_path: str,
    *,
    out_path: Optional[str] = None,
    sensitivity: Optional[float] = None,
    evalue: Optional[float] = None,
    max_seqs: Optional[int] = None,
    threads: Optional[int] = None,
    output_columns: Optional[Sequence[str]] = None,
    binary: Optional[str] = None,
) -> pd.DataFrame:
    """Run ``foldseek easy-search`` and return hits as a DataFrame.

    ``query_cif`` is a local mmCIF/PDB path; ``db_path`` is a foldseek
    target-database prefix (from :func:`download_database`) or a directory
    of structures. When ``out_path`` is given the raw m8 is written there
    durably (for provenance); otherwise it lands in a scratch dir that is
    cleaned up. All unset parameters fall through to foldseek's own
    defaults. Raises :class:`FoldseekError` on a non-zero exit.
    """
    resolved = resolve_foldseek_binary(binary)
    if not Path(query_cif).exists():
        raise FoldseekError(f"Query structure not found: {query_cif}")
    cols = list(output_columns or DEFAULT_OUTPUT_COLUMNS)

    scratch = Path(tempfile.mkdtemp(prefix="beak_foldseek_"))
    work = scratch / "tmp"
    if out_path is None:
        result_file = scratch / "result.m8"
    else:
        result_file = Path(out_path)
        result_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        resolved, "easy-search",
        str(query_cif), str(db_path), str(result_file), str(work),
        "--format-output", ",".join(cols),
    ]
    if sensitivity is not None:
        cmd += ["-s", str(sensitivity)]
    if evalue is not None:
        cmd += ["-e", str(evalue)]
    if max_seqs is not None:
        cmd += ["--max-seqs", str(max_seqs)]
    if threads is not None:
        cmd += ["--threads", str(threads)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            raise FoldseekError(
                f"foldseek easy-search failed (exit {proc.returncode})"
                + (f":\n{detail}" if detail else "")
            )
        text = result_file.read_text() if result_file.exists() else ""
        return parse_foldseek_m8(text, cols)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def download_database(
    db_name: str,
    dest_dir: str,
    *,
    binary: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    quiet: bool = False,
) -> Path:
    """Download a prebuilt foldseek database via ``foldseek databases``.

    Writes the database under ``dest_dir`` with a ``db`` prefix and
    returns that prefix path (the value to pass as ``db_path`` to
    :func:`run_easy_search`). By default foldseek's own download progress
    streams to the terminal; pass ``quiet=True`` to capture it instead.
    Raises :class:`FoldseekError` on a non-zero exit.
    """
    resolved = resolve_foldseek_binary(binary)
    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    db_prefix = dest / "db"

    cleanup = tmp_dir is None
    scratch = Path(tmp_dir).expanduser() if tmp_dir else Path(
        tempfile.mkdtemp(prefix="beak_foldseek_db_"))
    scratch.mkdir(parents=True, exist_ok=True)

    cmd = [resolved, "databases", db_name, str(db_prefix), str(scratch)]
    try:
        kwargs = {"text": True}
        if quiet:
            kwargs["capture_output"] = True
        proc = subprocess.run(cmd, **kwargs)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() if quiet else ""
            raise FoldseekError(
                f"foldseek databases {db_name} failed (exit {proc.returncode})"
                + (f":\n{detail}" if detail else "")
            )
        return db_prefix
    finally:
        if cleanup:
            shutil.rmtree(scratch, ignore_errors=True)
