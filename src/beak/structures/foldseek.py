"""Foldseek helpers shared between the remote job manager and the CLI.

Foldseek itself runs on the remote server (see ``beak.remote.foldseek``),
so this module holds only the format-level pieces both sides agree on: the
``easy-search`` output column set and a pure parser for its m8 output.
"""

from typing import List, Optional, Sequence

import pandas as pd


# Prebuilt databases foldseek can download via `foldseek databases`. Used to
# populate `beak setup foldseek` help and validate the requested database.
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

# Columns always requested from `foldseek easy-search --format-output`. Beyond
# the default m8 set we add alntmscore (global structural similarity, 0-1),
# lddt (local accuracy, 0-1), and query/target coverage — the fields worth
# landing on a structural-neighbours table. Order here IS the output order.
DEFAULT_OUTPUT_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "lddt", "qcov", "tcov",
]

# Column typing applied after the raw tab-split parse.
_FLOAT_COLS = {"fident", "evalue", "bits", "alntmscore", "lddt", "qcov", "tcov"}
_INT_COLS = {"alnlen", "mismatch", "gapopen", "qstart", "qend", "tstart", "tend"}


def parse_foldseek_m8(text: str,
                      columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Parse foldseek tab-separated m8 output into a typed DataFrame.

    Pure function — no subprocess, no SSH — so it is unit-testable against
    captured output. ``columns`` must match the ``--format-output`` used to
    produce ``text`` (defaults to :data:`DEFAULT_OUTPUT_COLUMNS`). Empty
    input yields an empty DataFrame with the right columns. Rows with too
    few/many fields are padded/truncated rather than dropped, so a stray
    column never desyncs the whole table.
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
