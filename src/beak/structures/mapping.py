"""Coordinate mapping between alignments, target sequences, and structures."""

import gemmi
import pandas as pd
from Bio.Align import PairwiseAligner, MultipleSeqAlignment


def map_alignment_to_target(alignment: MultipleSeqAlignment, target_id: str) -> pd.DataFrame:
    """Map alignment column indices to target sequence positions.

    Parameters
    ----------
    alignment : MultipleSeqAlignment
        The MSA containing the target sequence.
    target_id : str
        Record ID of the target sequence within the alignment.

    Returns
    -------
    pd.DataFrame
        Columns: aln_column (int, 0-based), target_pos (int, 1-based).
        Only includes columns where the target has a residue (not a gap).

    Raises
    ------
    KeyError
        If target_id is not found in the alignment.
    """
    # Find target record
    target_record = None
    for record in alignment:
        if record.id == target_id:
            target_record = record
            break

    if target_record is None:
        raise KeyError(f"Target ID '{target_id}' not found in alignment")

    target_seq = str(target_record.seq)

    rows = []
    target_pos = 0
    for col_idx, char in enumerate(target_seq):
        if char != '-':
            target_pos += 1
            rows.append({'aln_column': col_idx, 'target_pos': target_pos})

    return pd.DataFrame(rows)


def map_target_to_structure(
    target_seq: str,
    cif_path: str,
    chain_id: str = None,
) -> pd.DataFrame:
    """Align a target sequence to a structure and return position mapping.

    Parameters
    ----------
    target_seq : str
        Full target amino acid sequence (1-letter codes).
    cif_path : str
        Path to mmCIF file.
    chain_id : str, optional
        Chain to use. If None, uses first polymer chain.

    Returns
    -------
    pd.DataFrame
        Columns: target_pos (int, 1-based), pdb_resnum (int),
        pdb_resname (str, 1-letter).
        Only includes positions where both target and structure have residues.
    """
    struct_seq, residue_numbers, residue_names = _extract_sequence_from_chain(
        cif_path, chain_id
    )

    ref_aln, struct_aln = align_sequences(target_seq, struct_seq)

    # Walk alignment to build mapping (same logic as viz/structures.py:73-81)
    rows = []
    target_pos = 0
    struct_pos = 0
    for r_char, s_char in zip(ref_aln, struct_aln):
        if r_char != '-':
            target_pos += 1
        if s_char != '-':
            struct_pos += 1
        if r_char != '-' and s_char != '-':
            rows.append({
                'target_pos': target_pos,
                'pdb_resnum': residue_numbers[struct_pos - 1],
                'pdb_resname': residue_names[struct_pos - 1],
            })

    return pd.DataFrame(rows)


def _extract_sequence_from_chain(cif_path, chain_id=None):
    """Extract amino acid sequence from mmCIF chain.

    Returns
    -------
    tuple of (str, list[int], list[str])
        (sequence_str, residue_numbers, residue_names_1letter)
    """
    structure = gemmi.read_structure(cif_path)
    model = structure[0]

    if chain_id is not None:
        chain = model.find_chain(chain_id)
        if chain is None:
            raise ValueError(
                f"Chain '{chain_id}' not found in {cif_path}. "
                f"Available chains: {[c.name for c in model]}"
            )
    else:
        # Use first polymer chain
        chain = None
        for c in model:
            polymer = c.get_polymer()
            if len(polymer) > 0:
                chain = c
                break
        if chain is None:
            raise ValueError(f"No polymer chain found in {cif_path}")

    polymer = chain.get_polymer()

    sequence = []
    residue_numbers = []
    residue_names = []

    for residue in polymer:
        info = gemmi.find_tabulated_residue(residue.name)
        one_letter = info.one_letter_code if info.one_letter_code != '?' else 'X'
        sequence.append(one_letter)
        residue_numbers.append(residue.seqid.num)
        residue_names.append(one_letter)

    return ''.join(sequence), residue_numbers, residue_names


def align_sequences(seq_a, seq_b):
    """Global pairwise alignment of two sequences.

    Returns
    -------
    tuple of (str, str)
        (aligned_a, aligned_b) as strings with gap characters.
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(seq_a, seq_b)
    best = alignments[0]

    # Use indices array: shape (2, aln_length), -1 means gap
    indices = best.indices
    aligned_a = []
    aligned_b = []
    for i in range(indices.shape[1]):
        a_idx, b_idx = int(indices[0, i]), int(indices[1, i])
        aligned_a.append(seq_a[a_idx] if a_idx >= 0 else '-')
        aligned_b.append(seq_b[b_idx] if b_idx >= 0 else '-')

    return ''.join(aligned_a), ''.join(aligned_b)
