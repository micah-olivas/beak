"""PDB and AlphaFold structure discovery and download.

Uses PDBe SIFTS for UniProt → PDB mappings (batch POST endpoint) and
constructs AlphaFold DB URLs speculatively (no metadata API needed).
Downloads mmCIF files from RCSB and AlphaFold DB.
"""

import json
import os
import time
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional, Union


PDBE_SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"
ALPHAFOLD_CIF_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"
)
ALPHAFOLD_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

PDBE_BATCH_SIZE = 100
MAX_RETRIES = 3
RCSB_DELAY = 0.05
PDBE_DELAY = 0.2

METHOD_PRIORITY = {
    "X-ray diffraction": 1,
    "Electron Microscopy": 2,
    "Solution NMR": 3,
}


def resolve_alphafold_url(uniprot_id: str, fmt: str = "cif") -> str:
    """Resolve the canonical AlphaFold model URL via the prediction API.

    Robust to model-version bumps (the API returns whatever URL is current,
    so we don't break when v6 → v7). Falls back to a hardcoded v6 template
    only if the API itself is unreachable.

    Args:
        uniprot_id: UniProt accession.
        fmt: 'cif', 'pdb', or 'bcif'.

    Returns:
        Canonical model URL.

    Raises:
        FileNotFoundError: API definitively reports no model for this accession.
        ValueError: Unsupported `fmt`.
    """
    if fmt not in ('cif', 'pdb', 'bcif'):
        raise ValueError(f"Unsupported format: {fmt!r}")
    key = f"{fmt}Url"

    api_url = ALPHAFOLD_PREDICTION_API.format(uniprot_id=uniprot_id)
    try:
        with urllib.request.urlopen(api_url, timeout=10) as resp:
            data = json.loads(resp.read())
        if not data:
            raise FileNotFoundError(f"No AlphaFold model for {uniprot_id}")
        url = data[0].get(key)
        if url:
            return url
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"No AlphaFold model for {uniprot_id}"
            ) from e
        # Other HTTP errors → fall through to template fallback
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            json.JSONDecodeError):
        pass

    # Template fallback (network/parse error). Pinned to v6 — will need
    # bumping if v7 lands while AF is offline. Not worth chasing.
    if fmt == 'cif':
        return ALPHAFOLD_CIF_URL.format(uniprot_id=uniprot_id)
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.{fmt}"


def find_structures(
    uniprot_ids: List[str],
    source: str = "both",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Discover available structures for a list of UniProt IDs.

    Args:
        uniprot_ids: List of UniProt accession strings
        source: "pdb", "alphafold", or "both"
        on_progress: Optional callback(processed_count, total_count)

    Returns:
        DataFrame with columns: uniprot_id, source, structure_id, chain_id,
        resolution, method, coverage_start, coverage_end, download_url
    """
    unique_ids = list(dict.fromkeys(uniprot_ids))  # dedupe, preserve order
    rows = []
    processed = 0
    total = len(unique_ids)

    # PDB: batch POST to PDBe SIFTS
    if source in ("pdb", "both"):
        for i in range(0, len(unique_ids), PDBE_BATCH_SIZE):
            batch = unique_ids[i : i + PDBE_BATCH_SIZE]
            batch_rows = _fetch_pdb_mappings_batch(batch)
            rows.extend(batch_rows)
            processed = min(i + PDBE_BATCH_SIZE, total)
            if on_progress:
                on_progress(processed, total)
            if i + PDBE_BATCH_SIZE < len(unique_ids):
                time.sleep(PDBE_DELAY)

    # AlphaFold: generate speculative rows (no API call needed)
    if source in ("alphafold", "both"):
        for uid in unique_ids:
            rows.append({
                'uniprot_id': uid,
                'source': 'alphafold',
                'structure_id': f"AF-{uid}-F1",
                'chain_id': '-',
                'resolution': None,
                'method': 'AlphaFold prediction',
                'coverage_start': None,
                'coverage_end': None,
                'download_url': ALPHAFOLD_CIF_URL.format(uniprot_id=uid),
            })

    columns = [
        'uniprot_id', 'source', 'structure_id', 'chain_id',
        'resolution', 'method', 'coverage_start', 'coverage_end',
        'download_url',
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ['uniprot_id', 'source', 'resolution'],
        na_position='last',
    ).reset_index(drop=True)

    return df


def fetch_structures(
    structures_df: pd.DataFrame,
    output_dir: str = "structures",
    selection: str = "best",
    skip_existing: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Download structure files (mmCIF) to disk.

    Args:
        structures_df: DataFrame from find_structures()
        output_dir: Directory to save .cif files
        selection: "best" (one per UniProt ID per source), "all",
                   or an int N (top N per UniProt ID)
        skip_existing: Skip files that already exist on disk
        on_progress: Optional callback(downloaded_count, total_count)

    Returns:
        DataFrame with added 'local_path' column (path or None)
        and 'error' column for failed downloads.
    """
    if structures_df.empty:
        return structures_df.assign(local_path=None, error=None)

    # Apply selection
    df = _select_structures(structures_df, selection).copy()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    local_paths = []
    errors = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        filename = _make_filename(row)
        dest = os.path.join(output_dir, filename)

        if skip_existing and os.path.exists(dest):
            local_paths.append(dest)
            errors.append(None)
        else:
            try:
                _download_file(row['download_url'], dest)
                local_paths.append(dest)
                errors.append(None)
                time.sleep(RCSB_DELAY)
            except Exception as e:
                local_paths.append(None)
                errors.append(str(e))

        if on_progress:
            on_progress(idx + 1, total)

    df = df.copy()
    df['local_path'] = local_paths
    df['error'] = errors

    return df


# ── Private helpers ────────────────────────────────────────────


def _fetch_pdb_mappings_batch(uniprot_ids: List[str]) -> List[dict]:
    """POST a batch of UniProt IDs to PDBe SIFTS and parse the response."""
    data = ','.join(uniprot_ids).encode('utf-8')
    req = urllib.request.Request(
        PDBE_SIFTS_URL,
        data=data,
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        },
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return _parse_sifts_response(result)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_after)
                    continue
            if e.code == 404:
                # No mappings found for any ID in this batch
                return []
            raise
        except urllib.error.URLError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    return []


def _parse_sifts_response(result: dict) -> List[dict]:
    """Parse PDBe SIFTS JSON response into flat row dicts."""
    rows = []

    for uniprot_id, data in result.items():
        mappings = data.get('PDB', {})
        for pdb_id, chains in mappings.items():
            for chain_info in chains:
                chain_id = chain_info.get('chain_id', '')
                struct_asym = chain_info.get('struct_asym_id', chain_id)

                # Residue range
                start = chain_info.get('unp_start')
                end = chain_info.get('unp_end')

                # Resolution and method from experimental_method
                resolution = chain_info.get('resolution')
                method = chain_info.get('experimental_method', '')

                rows.append({
                    'uniprot_id': uniprot_id,
                    'source': 'pdb',
                    'structure_id': pdb_id.upper(),
                    'chain_id': chain_id,
                    'resolution': float(resolution) if resolution else None,
                    'method': method,
                    'coverage_start': int(start) if start else None,
                    'coverage_end': int(end) if end else None,
                    'download_url': f"{RCSB_DOWNLOAD_URL}/{pdb_id}.cif",
                })

    return rows


def _select_structures(
    df: pd.DataFrame,
    selection: Union[str, int],
) -> pd.DataFrame:
    """Apply selection strategy to reduce structures per UniProt ID."""
    if selection == "all":
        return df

    # Add score column for sorting
    scored = df.copy()
    scored['_score'] = scored.apply(_structure_score, axis=1)
    scored = scored.sort_values('_score').reset_index(drop=True)

    if selection == "best":
        # Best per (uniprot_id, source)
        result = scored.groupby(['uniprot_id', 'source']).first().reset_index()
    else:
        # Top N per uniprot_id
        n = int(selection)
        result = scored.groupby('uniprot_id').head(n).reset_index(drop=True)

    return result.drop(columns=['_score'], errors='ignore')


def _structure_score(row) -> tuple:
    """Scoring tuple for structure quality ranking. Lower is better."""
    method = row.get('method', '')
    method_rank = METHOD_PRIORITY.get(method, 99)

    resolution = row.get('resolution')
    res_val = resolution if resolution is not None else 999.0

    # Coverage span: larger is better, so negate
    start = row.get('coverage_start')
    end = row.get('coverage_end')
    if start is not None and end is not None:
        coverage = -(end - start)
    else:
        coverage = 0

    return (method_rank, res_val, coverage)


def _make_filename(row) -> str:
    """Generate a predictable filename from a structures DataFrame row."""
    uid = row['uniprot_id']
    source = row['source']

    if source == 'alphafold':
        return f"{uid}_AF.cif"
    else:
        struct_id = row['structure_id']
        chain = row.get('chain_id', '')
        if chain and chain != '-':
            return f"{uid}_{struct_id}_{chain}.cif"
        return f"{uid}_{struct_id}.cif"


def _download_file(url: str, dest_path: str) -> None:
    """Download a single file with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(url, dest_path)
            return
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_after)
                    continue
            if e.code == 404:
                raise FileNotFoundError(
                    f"Structure not found: {url}"
                ) from e
            raise
        except urllib.error.URLError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
