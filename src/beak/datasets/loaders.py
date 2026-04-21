"""Functions to load bundled reference datasets."""

import json
import urllib.request
import urllib.error
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

GROWTH_TEMPS_FILE = "enqvist_growth_temps.tsv"
GROWTH_TEMPS_URL = "https://zenodo.org/records/1175609/files/temperature_data.tsv?download=1"


def download_growth_temps(force: bool = False) -> Path:
    """Download the Enqvist growth-temperature dataset from Zenodo.

    Args:
        force: Re-download even if the file already exists locally.

    Returns:
        Path to the local TSV file.

    Reference:
        Enqvist, M. et al. (2018). Growth temperatures of prokaryotes.
        Zenodo. https://doi.org/10.5281/zenodo.1175609
    """
    dest = DATA_DIR / GROWTH_TEMPS_FILE

    if dest.exists() and not force:
        return dest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading growth-temperature dataset (~5 MB) from Zenodo...")
    urllib.request.urlretrieve(GROWTH_TEMPS_URL, dest)
    print(f"✓ Saved to {dest}")
    return dest


def load_growth_temps() -> pd.DataFrame:
    """Load the Enqvist microbial growth-temperature dataset.

    Auto-downloads from Zenodo on first call (~5 MB, cached locally).

    Returns:
        DataFrame with columns:
            - organism: organism name (lowercased, underscore-joined)
            - domain: Bacteria / Archaea / Eukaryota
            - temperature: optimal growth temperature (°C)
            - taxid: NCBI taxonomy ID
            - lineage_text: full lineage string
            - superkingdom, phylum, class, order, family, genus: taxid per level

    Reference:
        Enqvist, M. et al. (2018). Growth temperatures of prokaryotes.
        Zenodo. https://doi.org/10.5281/zenodo.1175609

    Example:
        >>> temps = load_growth_temps()
        >>> temps[temps['organism'].str.contains('escherichia')]
    """
    data_file = download_growth_temps()
    return pd.read_csv(data_file, sep='\t')


def list_datasets() -> pd.DataFrame:
    """List all available bundled datasets."""
    metadata_file = DATA_DIR / "metadata.json"

    if not metadata_file.exists():
        return pd.DataFrame([
            {
                'name': 'enqvist_growth_temps',
                'description': 'Microbial growth temperatures',
                'source': 'https://zenodo.org/records/1175609',
                'loader': 'load_growth_temps()'
            }
        ])

    with open(metadata_file) as f:
        meta = json.load(f)

    return pd.DataFrame(meta['datasets'])
