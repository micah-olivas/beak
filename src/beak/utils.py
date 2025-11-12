"""Utility functions for local sequence analysis"""

import pandas as pd
from typing import Optional
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def parse_uniprot_header(header: str) -> dict:
    """
    Parse UniProt FASTA header into components
    
    Args:
        header: UniProt header (e.g., 'sp|P12345|PROT_HUMAN ...')
    
    Returns:
        Dict with 'db', 'accession', 'name', 'description'
    
    Example:
        >>> parse_uniprot_header('sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein')
        {'db': 'sp', 'accession': 'P0DTC2', 'name': 'SPIKE_SARS2', 
         'description': 'Spike glycoprotein'}
    """
    parts = header.split('|')
    
    if len(parts) >= 3:
        db = parts[0]
        accession = parts[1]
        rest = parts[2].split(' ', 1)
        name = rest[0]
        description = rest[1] if len(rest) > 1 else ''
        
        return {
            'db': db,
            'accession': accession,
            'name': name,
            'description': description
        }
    else:
        return {
            'db': None,
            'accession': None,
            'name': header,
            'description': ''
        }


def parse_fasta_headers(
    df: pd.DataFrame,
    header_col: str = 'target',
    parse_uniprot: bool = True
) -> pd.DataFrame:
    """
    Extract information from FASTA headers
    
    Args:
        df: DataFrame with FASTA headers
        header_col: Column containing headers
        parse_uniprot: If True, parse UniProt-style headers
    
    Returns:
        DataFrame with additional columns extracted from headers
    """
    if parse_uniprot:
        parsed = df[header_col].apply(parse_uniprot_header)
        parsed_df = pd.DataFrame(parsed.tolist())
        return pd.concat([df, parsed_df], axis=1)
    else:
        # Simple split on first space
        df['header_id'] = df[header_col].str.split(' ').str[0]
        df['header_description'] = df[header_col].str.split(' ', 1).str[1]
        return df


def add_sequence_properties(
    df: pd.DataFrame,
    sequence_col: str = 'sequence'
) -> pd.DataFrame:
    """
    Add computed sequence properties
    
    Args:
        df: DataFrame with sequences
        sequence_col: Column containing protein sequences
    
    Returns:
        DataFrame with added properties: length, molecular_weight, 
        isoelectric_point, aromaticity, instability_index
    
    Example:
        >>> df = pd.DataFrame({'sequence': ['MKTAYIAK', 'ACDEFGH']})
        >>> add_sequence_properties(df)
    """
    def compute_properties(seq):
        if pd.isna(seq) or len(seq) == 0:
            return pd.Series({
                'length': 0,
                'molecular_weight': 0,
                'isoelectric_point': 0,
                'aromaticity': 0,
                'instability_index': 0
            })
        
        try:
            analyzer = ProteinAnalysis(str(seq))
            return pd.Series({
                'length': len(seq),
                'molecular_weight': analyzer.molecular_weight(),
                'isoelectric_point': analyzer.isoelectric_point(),
                'aromaticity': analyzer.aromaticity(),
                'instability_index': analyzer.instability_index()
            })
        except Exception as e:
            # Return zeros if analysis fails
            return pd.Series({
                'length': len(seq),
                'molecular_weight': 0,
                'isoelectric_point': 0,
                'aromaticity': 0,
                'instability_index': 0
            })
    
    properties = df[sequence_col].apply(compute_properties)
    return pd.concat([df, properties], axis=1)


def annotate_temperature(
    df: pd.DataFrame,
    organism_col: str = 'scientific_name',
    method: str = 'exact'
) -> pd.DataFrame:
    """
    Annotate sequences with growth temperatures from Enqvist dataset
    
    Args:
        df: DataFrame with organism names
        organism_col: Column containing organism names
        method: 'exact' or 'fuzzy' matching
    
    Returns:
        DataFrame with added 'growth_temp' and 'temp_source' columns
    
    Example:
        >>> tax_results = taxonomy.get_results(job_id)
        >>> annotated = annotate_temperature(tax_results)
        >>> annotated[['scientific_name', 'growth_temp']].head()
    """
    from .datasets import load_growth_temps
    
    temps = load_growth_temps()
    
    if method == 'exact':
        # Exact matching
        merged = df.merge(
            temps[['organism', 'temperature', 'domain']],
            left_on=organism_col,
            right_on='organism',
            how='left'
        )
        merged = merged.rename(columns={
            'temperature': 'growth_temp',
            'domain': 'temp_domain'
        })
        merged = merged.drop(columns=['organism'], errors='ignore')
        
    elif method == 'fuzzy':
        # Fuzzy matching for partial names
        try:
            from fuzzywuzzy import process
        except ImportError:
            raise ImportError(
                "Fuzzy matching requires 'fuzzywuzzy'. "
                "Install with: pip install fuzzywuzzy python-Levenshtein"
            )
        
        def find_best_match(name):
            if pd.isna(name):
                return None, 0
            match = process.extractOne(name, temps['organism'].tolist())
            return match[0] if match else None, match[1] if match else 0
        
        matches = df[organism_col].apply(find_best_match)
        df['matched_organism'] = [m[0] for m in matches]
        df['match_score'] = [m[1] for m in matches]
        
        merged = df.merge(
            temps[['organism', 'temperature', 'domain']],
            left_on='matched_organism',
            right_on='organism',
            how='left'
        )
        merged = merged.rename(columns={
            'temperature': 'growth_temp',
            'domain': 'temp_domain'
        })
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'fuzzy'")
    
    return merged


def filter_by_temperature(
    df: pd.DataFrame,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    temp_col: str = 'growth_temp'
) -> pd.DataFrame:
    """
    Filter sequences by growth temperature range
    
    Args:
        df: DataFrame with temperature annotations
        min_temp: Minimum temperature (inclusive)
        max_temp: Maximum temperature (inclusive)
        temp_col: Column containing temperatures
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> # Get only thermophiles (>50°C)
        >>> thermophiles = filter_by_temperature(df, min_temp=50)
    """
    filtered = df.copy()
    
    if min_temp is not None:
        filtered = filtered[filtered[temp_col] >= min_temp]
    
    if max_temp is not None:
        filtered = filtered[filtered[temp_col] <= max_temp]
    
    return filtered