"""Parsing and property computation for protein sequences and FASTA records."""

import warnings

import numpy as np
import pandas as pd
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
        df['header_id'] = df[header_col].str.split(' ').str[0]
        df['header_description'] = df[header_col].str.split(' ', n=1).str[1]
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
    nan_properties = pd.Series({
        'length': 0,
        'molecular_weight': np.nan,
        'isoelectric_point': np.nan,
        'aromaticity': np.nan,
        'instability_index': np.nan,
    })

    def compute_properties(seq):
        if pd.isna(seq) or len(seq) == 0:
            return nan_properties

        try:
            analyzer = ProteinAnalysis(str(seq))
            return pd.Series({
                'length': len(seq),
                'molecular_weight': analyzer.molecular_weight(),
                'isoelectric_point': analyzer.isoelectric_point(),
                'aromaticity': analyzer.aromaticity(),
                'instability_index': analyzer.instability_index(),
            })
        except Exception as exc:
            warnings.warn(
                f"ProteinAnalysis failed on sequence ({type(exc).__name__}: {exc}); "
                f"returning NaN properties.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.Series({**nan_properties.to_dict(), 'length': len(seq)})

    properties = df[sequence_col].apply(compute_properties)
    return pd.concat([df, properties], axis=1)
