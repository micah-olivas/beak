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


# PDBe SIFTS UniProt → PDB mapping endpoint. As of 2026 the
# `/mappings/uniprot` POST batch route is gone (returns 405); we
# now hit `/mappings/best_structures/<uid>` which returns per-chain
# entries already pre-ranked by experimental quality (lower
# resolution X-ray first, NMR / EM following) AND populates the
# `resolution` + `experimental_method` fields the local ranker uses
# downstream. The plain `/mappings/<uid>` endpoint also exists but
# omits resolution/method, so the local "best" picker can't tell
# the structures apart.
PDBE_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"
ALPHAFOLD_CIF_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"
)
ALPHAFOLD_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

PDBE_BATCH_SIZE = 100
MAX_RETRIES = 3
RCSB_DELAY = 0.05
PDBE_DELAY = 0.2
# Per-request socket timeouts. Without these, a stalled PDBe / RCSB /
# AlphaFold endpoint hangs the calling worker forever and the UI screen
# can't recover. Downloads get a longer budget than metadata fetches
# because CIFs of >5k-residue assemblies can be 10+ MB on slow links.
HTTP_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120

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
    """Fetch UniProt → PDB mappings from PDBe SIFTS.

    PDBe deprecated the POST-with-comma-separated-ids batch endpoint
    (it now returns 405 Method Not Allowed; verified live 2026-05).
    The replacement is a per-UniProt GET against
    ``/pdbe/api/mappings/<id>`` — same JSON response shape as the
    old POST batch, so the parser is reused unchanged. We loop the
    incoming list and merge the per-id results, preserving the
    historical "batch" return contract for callers.

    Each per-id request goes through `_get_sifts_for_uniprot`, which
    handles the retry-on-transient and 404-as-empty cases.
    """
    rows: List[dict] = []
    for uid in uniprot_ids:
        rows.extend(_get_sifts_for_uniprot(uid))
        # Small spacing between consecutive GETs so a project with
        # multiple targets doesn't hammer PDBe; matches the
        # `PDBE_DELAY` cadence the old batch path used between
        # batches.
        if len(uniprot_ids) > 1:
            time.sleep(PDBE_DELAY)
    return rows


def _get_sifts_for_uniprot(uniprot_id: str) -> List[dict]:
    """GET pre-ranked PDB structures for a single UniProt id.

    Hits the `/mappings/best_structures/<id>` endpoint, which:
      * returns per-chain entries already sorted by experimental
        quality (resolution-then-coverage), so callers that just
        want the top hit can take ``rows[0]``;
      * carries the `resolution` and `experimental_method` fields
        so the local `_select_structures("best")` ranker stays
        meaningful when callers want a different secondary sort.
    """
    url = f"{PDBE_BEST_STRUCTURES_URL}/{uniprot_id}"
    req = urllib.request.Request(
        url,
        headers={'Accept': 'application/json'},
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
                result = json.loads(response.read().decode('utf-8'))
                return _parse_best_structures_response(result)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_after)
                    continue
            if e.code == 404:
                # No PDB structures for this UniProt — empty result,
                # not an error. Caller should fall back to AlphaFold.
                return []
            raise
        except urllib.error.URLError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    return []


def _parse_best_structures_response(result: dict) -> List[dict]:
    """Parse `/mappings/best_structures/<id>` JSON into flat row dicts.

    Response shape::

        {"<uniprot_id>": [
            {"pdb_id": "...", "chain_id": "A", "resolution": 1.45,
             "experimental_method": "X-ray diffraction",
             "unp_start": 1, "unp_end": 99, "coverage": 1.0, ...},
            ...
        ]}

    Different from the older `/mappings/<id>` shape (which nested
    chains under `{"PDB": {<pdb_id>: [...]}}` and omitted resolution).
    """
    rows = []
    for uniprot_id, entries in result.items():
        for entry in entries:
            pdb_id = entry.get('pdb_id') or ''
            if not pdb_id:
                continue
            resolution = entry.get('resolution')
            start = entry.get('unp_start')
            end = entry.get('unp_end')
            rows.append({
                'uniprot_id': uniprot_id,
                'source': 'pdb',
                'structure_id': pdb_id.upper(),
                'chain_id': entry.get('chain_id', ''),
                'resolution': float(resolution) if resolution is not None else None,
                'method': entry.get('experimental_method', ''),
                'coverage_start': int(start) if start is not None else None,
                'coverage_end': int(end) if end is not None else None,
                'download_url': f"{RCSB_DOWNLOAD_URL}/{pdb_id}.cif",
            })
    return rows


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
    """Download a single file with retry logic.

    Streams the response body to disk via shutil.copyfileobj so we can
    enforce a socket timeout (urlretrieve doesn't expose one) and so
    we can delete a half-written file on any mid-transfer failure —
    otherwise the next load would happily parse the corrupt CIF.
    """
    import shutil
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp:
                with open(dest_path, 'wb') as out:
                    shutil.copyfileobj(resp, out)
            return
        except urllib.error.HTTPError as e:
            # Partial file may exist from an earlier successful header
            # read followed by a body failure — purge before retrying
            # so a 429-then-success doesn't keep stale bytes around.
            _unlink_if_exists(dest_path)
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
        except (urllib.error.URLError, TimeoutError, OSError):
            _unlink_if_exists(dest_path)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def _unlink_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        # Best-effort cleanup — caller is already handling the
        # original network error and shouldn't be derailed by an
        # unlink failure on the partial file.
        pass
