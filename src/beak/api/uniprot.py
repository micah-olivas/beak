"""UniProt REST API client.

Pfam-based protein lookups and single-accession FASTA retrieval.
Uses urllib.request (no external dependencies).
"""

import re
import json
import time
import tempfile
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

MAX_RETRIES = 3
DEFAULT_PAGE_SIZE = 500


def fetch_uniprot(accession: str, output_dir: Optional[str] = None) -> str:
    """
    Fetch a protein sequence from UniProt by accession ID.

    Args:
        accession: UniProt accession (e.g., 'P0DTC2')
        output_dir: Directory to save the FASTA file. If None, uses a temp dir.

    Returns:
        Path to the downloaded FASTA file
    """
    url = UNIPROT_FASTA_URL.format(accession=accession)

    try:
        with urllib.request.urlopen(url) as response:
            fasta_content = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            raise ValueError(f"UniProt accession '{accession}' not found") from e
        raise

    if not fasta_content.startswith('>'):
        raise ValueError(f"UniProt accession '{accession}' not found or returned invalid data")

    if output_dir:
        out_path = Path(output_dir) / f"{accession}.fasta"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(tempfile.mkdtemp()) / f"{accession}.fasta"

    out_path.write_text(fasta_content)
    return str(out_path)


def query_uniprot_by_pfam(
    pfam_id: str,
    taxonomy: bool = False,
    lineage: bool = False,
    size: int = DEFAULT_PAGE_SIZE,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Query UniProt for all proteins containing a Pfam domain.

    Args:
        pfam_id: Pfam accession (e.g., "PF00069")
        taxonomy: If True, include organism_name and organism_id
        lineage: If True, include full taxonomic lineage (implies taxonomy)
        size: Page size for pagination (max 500)
        on_progress: Optional callback(fetched_count, page_number) called
                     after each page is fetched

    Returns:
        DataFrame with columns depending on flags:
            - always: accession, protein_name
            - taxonomy: + organism_name, organism_id
            - lineage: + lineage
    """
    fields = ['accession', 'protein_name']
    if lineage or taxonomy:
        fields.extend(['organism_name', 'organism_id'])
    if lineage:
        fields.append('lineage')

    url = _build_query_url(pfam_id, fields, min(size, 500))
    all_results = []
    page = 0

    while url:
        results, url = _fetch_page(url)
        all_results.extend(results)
        page += 1
        if on_progress:
            on_progress(len(all_results), page)

    if not all_results:
        return pd.DataFrame(columns=fields)

    rows = []
    for entry in all_results:
        row = {
            'accession': entry.get('primaryAccession', ''),
            'protein_name': _extract_protein_name(entry),
        }
        if taxonomy or lineage:
            organism = entry.get('organism', {})
            row['organism_name'] = organism.get('scientificName', '')
            row['organism_id'] = organism.get('taxonId', '')
        if lineage:
            lineage_list = entry.get('organism', {}).get('lineage', [])
            row['lineage'] = ' > '.join(lineage_list)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_query_url(
    pfam_id: str,
    fields: List[str],
    size: int,
    cursor: Optional[str] = None,
) -> str:
    """Build the UniProt search URL with query parameters."""
    params = {
        'query': f'(xref:pfam-{pfam_id})',
        'fields': ','.join(fields),
        'size': str(size),
        'format': 'json',
    }
    if cursor:
        params['cursor'] = cursor

    return f"{UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"


def _fetch_page(url: str) -> tuple:
    """Fetch one page of UniProt results.

    Args:
        url: Full URL to fetch

    Returns:
        (results_list, next_url_or_None)
    """
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                next_url = _parse_link_header(response.headers.get('Link', ''))
                return data.get('results', []), next_url

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — respect Retry-After header
                retry_after = int(e.headers.get('Retry-After', 5))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_after)
                    continue
            raise

    return [], None


def _parse_link_header(link_header: str) -> Optional[str]:
    """Extract the 'next' URL from a Link response header.

    Format: <https://rest.uniprot.org/...?cursor=xxx>; rel="next"
    """
    if not link_header:
        return None

    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _extract_protein_name(entry: dict) -> str:
    """Extract the recommended protein name from a UniProt entry."""
    protein = entry.get('proteinDescription', {})

    rec_name = protein.get('recommendedName', {})
    if rec_name:
        full_name = rec_name.get('fullName', {})
        if isinstance(full_name, dict):
            return full_name.get('value', '')
        return str(full_name)

    sub_names = protein.get('submissionNames', [])
    if sub_names:
        full_name = sub_names[0].get('fullName', {})
        if isinstance(full_name, dict):
            return full_name.get('value', '')
        return str(full_name)

    return ''
