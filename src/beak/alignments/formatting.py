import numpy as np
import pandas as pd
from tqdm import tqdm
from pyparsing import Optional

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

def subsample_aln(
    alignment: MultipleSeqAlignment,
    n_sequences: int,
    random_state: int = None,
    keep_records: list = None,
) -> MultipleSeqAlignment:
    """
    Subsample sequences from a multiple sequence alignment.

    Args:
        alignment: MultipleSeqAlignment object
        n_sequences: Number of sequences to sample
        random_state: Seed for reproducibility
        keep_records: List of record IDs to always include in the sample
    Returns:
        Subsampled MultipleSeqAlignment
    """
    import random

    if random_state is not None:
        random.seed(random_state)

    total_seqs = len(alignment)
    n_sequences = min(n_sequences, total_seqs)

    # If keep_records is specified, ensure they are included
    if keep_records is not None:
        keep_set = set(keep_records)
        kept_records = [record for record in alignment if record.id in keep_set]
        n_to_sample = n_sequences - len(kept_records)

        if n_to_sample <= 0:
            # If we already have enough kept records, just return those
            return MultipleSeqAlignment(kept_records[:n_sequences])

        # Sample from remaining records
        remaining_records = [record for record in alignment if record.id not in keep_set]
        sampled_records = random.sample(remaining_records, n_to_sample)
        final_records = kept_records + sampled_records
    else:
        # Simple random sampling
        final_records = random.sample(list(alignment), n_sequences)

    return MultipleSeqAlignment(final_records)

def ungap(
    alignment: MultipleSeqAlignment,
    threshold: float = 1.0,
) -> MultipleSeqAlignment:
    """
    Remove gap columns from a multiple sequence alignment.

    Args:
        alignment: MultipleSeqAlignment object
        threshold: Proportion of sequences that must have a gap
                   in a column for it to be removed (0.0 to 1.0)
    Returns:
        MultipleSeqAlignment without gap columns
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    n_seqs = len(alignment)

    # Convert alignment to numpy array for vectorized operations
    aln_array = np.array([list(str(record.seq)) for record in alignment])

    # Determine columns to keep based on gap threshold (vectorized)
    gap_counts = (aln_array == '-').sum(axis=0)
    keep_columns = gap_counts < (threshold * n_seqs)

    # Filter columns (vectorized boolean indexing)
    ungapped_array = aln_array[:, keep_columns]

    # Convert back to strings efficiently using numpy array operations
    ungapped_seqs = [''.join(row) for row in ungapped_array]

    # Create new alignment preserving record metadata
    ungapped_records = [
        SeqRecord(
            Seq(seq),
            id=record.id,
            name=record.name,
            description=record.description
        )
        for record, seq in tqdm(zip(alignment, ungapped_seqs), total=len(ungapped_seqs), desc="Reconstructing ungapped records")
    ]

    return MultipleSeqAlignment(ungapped_records)

def aln_to_pssm(
    alignment: MultipleSeqAlignment,
    as_freq: bool = False,
    include_unknown: bool = True,
) -> pd.DataFrame:
    """
    Generate Position-Specific Scoring Matrix (PSSM)
    from a multiple sequence alignment.

    Args:
        alignment: MultipleSeqAlignment object
        as_freq: If True, return frequencies instead of counts
        include_unknown: If True, include counts for unknown amino acids (X, B, Z, etc.)
    Returns:
        DataFrame with positions as rows and amino acids as columns
    """
    # Set alphabet - standard 20 amino acids plus gap
    alphabet = 'ACDEFGHIKLMNPQRSTVWY-'

    # Add unknown character if requested
    if include_unknown:
        alphabet += 'X'

    # Get alignment dimensions
    aln_length = alignment.get_alignment_length()

    # Convert alignment to numpy array for vectorized operations
    aln_array = np.array([list(str(record.seq).upper()) for record in alignment])

    # Replace non-standard amino acids with 'X' if include_unknown is True
    if include_unknown:
        # Any character not in the standard alphabet becomes 'X'
        standard_chars = set('ACDEFGHIKLMNPQRSTVWY-')
        mask = ~np.isin(aln_array, list(standard_chars))
        aln_array[mask] = 'X'

    # Initialize PSSM matrix (positions × amino acids)
    pssm = np.zeros((aln_length, len(alphabet)), dtype=int)

    # Vectorized counting: for each amino acid, count across all sequences
    for i, aa in tqdm(enumerate(alphabet), total=len(alphabet), desc="Calculating PSSM"):
        pssm[:, i] = (aln_array == aa).sum(axis=0)

    # Convert to DataFrame with positions as index, amino acids as columns
    pssm_df = pd.DataFrame(pssm,
                           index=range(aln_length),
                           columns=list(alphabet))

    # Convert counts to frequencies if requested
    if as_freq:
        pssm_df = pssm_df.div(pssm_df.sum(axis=1), axis=0)

    return pssm_df

def aln_to_consensus(
    alignment: MultipleSeqAlignment,
    threshold: float = 0,
    gap_char: str = '-',
    unknown_char: str = 'X',
    plurality: bool = False
) -> str:
    """
    Generate consensus sequence from a multiple sequence alignment.

    Args:
        alignment: MultipleSeqAlignment object
        threshold: Minimum frequency for an amino acid to be included in consensus
        gap_char: Character representing gaps in the alignment
        unknown_char: Character representing unknown amino acids
        plurality: If True, use most common amino acid regardless of threshold
    Returns:
        Consensus sequence as a string
    """
    # Generate PSSM as frequencies
    pssm = aln_to_pssm(alignment, as_freq=True, include_unknown=True)

    consensus_chars = []
    for pos in tqdm(pssm.index, desc="Generating consensus sequence"):
        freqs = pssm.loc[pos]

        # Exclude gaps and unknowns from consensus selection
        freqs_no_gap = freqs#.drop(['-', 'X'], errors='ignore')

        if len(freqs_no_gap) == 0 or freqs_no_gap.max() == 0:
            # All gaps/unknowns at this position
            consensus_chars.append(gap_char)
            continue

        max_aa = freqs_no_gap.idxmax()
        max_freq = freqs_no_gap.max()

        if plurality or max_freq >= threshold:
            consensus_chars.append(max_aa)
        else:
            consensus_chars.append(unknown_char)

    return ''.join(consensus_chars)