"""Functions to load bundled reference datasets"""

import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent / "data"


def load_growth_temps() -> pd.DataFrame:
    """
    Load Enqvist et al. microbial growth temperature dataset
    
    Returns:
        DataFrame with columns:
            - organism: Organism name
            - temperature: Optimal growth temperature (°C)
            - domain: Taxonomic domain
            - source: Data source
    
    Reference:
        Enqvist, M. et al. (2018). Growth temperatures of prokaryotes.
        Zenodo. https://doi.org/10.5281/zenodo.1175609
    
    Example:
        >>> temps = load_growth_temps()
        >>> temps[temps['organism'].str.contains('Escherichia')]
    """
    data_file = DATA_DIR / "enqvist_growth_temps.tsv"
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Growth temperature dataset not found at {data_file}\n"
            f"Please ensure the dataset is downloaded to beak/datasets/data/"
        )
    
    return pd.read_csv(data_file, sep='\t')


def list_datasets() -> pd.DataFrame:
    """
    List all available bundled datasets
    
    Returns:
        DataFrame with dataset information
    """
    metadata_file = DATA_DIR / "metadata.json"
    
    if not metadata_file.exists():
        # Return basic info if metadata doesn't exist
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