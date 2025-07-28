import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import Bio
from Bio import AlignIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')

import seqlogo

def aln_to_dict(aln):
    """Converts alignment to a dictionary.

    Args:
        aln (Bio.Align.MultipleSeqAlignment): Alignment object

    Returns:
        dict: Dictionary with sequence IDs as keys and sequences as values
    """
    return {record.id: str(record.seq) for record in aln}

def aln_to_df(aln):
    """Converts alignment to Pandas DataFrame

    Args:
        aln (Bio.Align.MultipleSeqAlignment): Alignment object

    Returns:
        pd.DataFrame: DataFrame with sequence IDs as index and positions as columns
    """
    data = {record.id: str(record.seq) for record in aln}
    df = pd.DataFrame.from_dict(data, orient='index', columns=['sequence'])
    return df

def ungap_aln(aln, ref_seq=None, gap_threshold=0.3):
    """
    Remove gapped positions from a multiple sequence alignment.

    Parameters:
        aln (Bio.Align.MultipleSeqAlignment): Alignment object from Biopython.
        gap_threshold (float): Proportion of gaps to consider a column mostly gaps (default 0.5).

    Returns:
        Bio.Align.MultipleSeqAlignment: New alignment object with gapped columns removed.
    """

    # Get sequence IDs and sequences from the alignment
    sequence_ids = [record.id for record in aln]
    sequences = [str(record.seq) for record in aln]

    # Transpose alignment to work on columns
    columns = list(zip(*sequences))

    # Identify columns to keep (those with gaps below the threshold)
    keep_columns = [
        i for i, column in enumerate(columns)
        if column.count('-') / len(column) <= gap_threshold
    ]

    if ref_seq != None:
        keep_columns = [
            i for i, column in enumerate(columns)
            if (ref_seq[i] != "-")
        ]


    # Filter sequences to retain only the columns to keep
    ungapped_records = [
        SeqRecord(Seq(''.join(seq[i] for i in keep_columns)), id=seq_id, description="")
        for seq_id, seq in zip(sequence_ids, sequences)
    ]

    return MultipleSeqAlignment(ungapped_records)

def get_consensus(alignment):
    """Generate a naive consensus from a fasta alignment

    Args:
        alignment (Bio.Align.MultipleSeqAlignment): alignment

    Returns:
        str: consensus sequence
    """
    consensus = ""
    for i in range(alignment.get_alignment_length()):
        counts = {}
        for record in alignment:
            AA = record.seq[i]
            # print(AA)
            if AA in counts:
                counts[AA] += 1
            else:
                counts[AA] = 1
        consensus += max(counts, key=counts.get)
    return consensus

def conservation(alignment):
    """Compute positional conservation from alignment.

    Args:
        alignment (Bio.Align.MultipleSeqAlignment): Bio alignment object

    Returns:
        np.array: array of conservation values matching the aligned sequence length
    """
    aln_len = alignment.get_alignment_length()
    conservation = np.zeros(aln_len)
    for i in range(aln_len):
        counts = {}
        for record in alignment:
            AA = record.seq[i]
            if AA in counts:
                counts[AA] += 1
            else:
                counts[AA] = 1
        total = sum(counts.values())
        freqs = np.array([v / total for v in counts.values()])
        # Shannon entropy
        entropy = -np.sum(freqs * np.log2(freqs + 1e-12))
        # Conservation as 1 - normalized entropy
        max_entropy = np.log2(len(counts))
        conservation[i] = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
    return conservation

def alignment_to_pssm(alignment, freq=False):
    """Generate a position-specific scoring matrix (PSSM) from a Bio alignment

    Args:
        alignment (Bio.Align.MultipleSeqAlignment): Bio alignment object
        freq (bool): If True, return frequencies instead of counts.

    Returns:
        pd.DataFrame: position-specific scoring matrix
    """

    # Amino acid order (including gap)
    amino_acids = ["A", "R", "N", "D", "C", "Q", \
                   "E", "G", "H", "I", "L", "K", "M", \
                    "F", "P", "S", "T", "W", "Y", "V", "-"]
    data = []
    for i in range(alignment.get_alignment_length()):
        row = []
        counts = {}
        for record in alignment:
            AA = record.seq[i]
            counts[AA] = counts.get(AA, 0) + 1
        for AA in amino_acids:
            count = counts.get(AA, 0)
            if freq:
                count = count / len(alignment)
            row.append(count)
        data.append(row)
        
    # Index: positions, Columns: amino acids
    pssm_df = pd.DataFrame(data, columns=amino_acids)

    # Compute conservation
    pssm_df['cons_i'] = conservation(alignment)
    return pssm_df

def single_sequence_aln_frequencies(query_seq, pssm, check_positions=False, consensus_seq=None):
    """
    For a query sequence, get the frequency of each residue from the pssm.

    Args:
        query_seq (str): The query sequence (same length as pssm columns, or will be aligned if check_positions).
        pssm (pd.DataFrame): Position-specific scoring matrix (frequencies or counts).
        check_positions (bool): If True, pairwise align query_seq to consensus_seq to determine correct positions.
        consensus_seq (str, optional): Consensus sequence to align to if check_positions is True.

    Returns:
        np.ndarray: 1D array of frequencies for each residue in the query sequence.
    """
    # Get consensus from pssm: for each position (row), pick the column (AA) with max value
    consensus_from_pssm = ''.join(pssm.idxmax(axis=1))

    if check_positions:
        # Use consensus from pssm, ignore consensus_seq argument
        aln = pairwise2.align.globalms(query_seq, consensus_from_pssm, 2, -1, -5, -0.5, one_alignment_only=True)[0]
        aligned_consensus, aligned_query = aln.seqA, aln.seqB

        # print("Aligned query seq:\t", aligned_query)
        # print("Aligned consensus:\t", aligned_consensus)

        freqs = []
        pssm_pos = 0
        for c_aa, q_aa in zip(aligned_consensus, aligned_query):
            if c_aa != '-':
                if q_aa in pssm.columns and pssm_pos in pssm.index:
                    # print(f"Freq at {pssm_pos}{q_aa}: ", pssm.at[pssm_pos, q_aa])
                    freq = pssm.at[pssm_pos, q_aa]
                else:
                    freq = 0.0
                freqs.append(freq)
                pssm_pos += 1
            else:
                # gap in consensus, skip position
                continue

        return np.array(freqs).reshape(1, len(freqs))
    else:
        freqs = []
        for i, aa in enumerate(query_seq):
            if i in pssm.index and aa in pssm.columns:
                # print(f"Freq at {i}{aa}: ", pssm.at[i, aa])
                freqs.append(pssm.at[i, aa])
            else:
                freqs.append(0.0)
        return np.array(freqs).reshape(1, len(freqs))

def plot_sequence_logo(pssm, title="Sequence Logo"):
    """
    Plot a sequence logo from a PSSM using the seqlogo package.

    Args:
        pssm (pd.DataFrame): Position-specific scoring matrix (frequencies, not counts).
        title (str): Title for the plot.
    """
    # Ensure pssm columns are positions and rows are amino acids
    # seqlogo expects a pandas DataFrame with index as letters and columns as positions
    # and values as probabilities (frequencies)
    if pssm.values.max() > 1.0:
        # Convert counts to frequencies
        pssm = pssm.div(pssm.sum(axis=0), axis=1)

    # Specify the protein alphabet for seqlogo
    protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    # Filter the PSSM to only include standard amino acids in the correct order
    pssm = pssm.reindex(list(protein_alphabet))
    pwm = seqlogo.Pwm(pssm)
    seqlogo.seqlogo(pwm, ic_scale=True, format='png', size='large', title=title)

def pssms_by_taxon(seq_tax_df, rank):
    """Generate a dictionary of PSSMs for each taxon of a defined rank
    """
    # Group by superkingdom and get all aligned sequences as lists
    rank_grouped_seqs = seq_tax_df.groupby(rank)['Aligned_sequence'].apply(list)

    print(f"Found {len(rank_grouped_seqs)} values of {rank}:")
    for sk, seqs in rank_grouped_seqs.items():
        print(f"  {sk}: {len(seqs)} sequences")

    pssms_tax_dict = {}

    for sk, seqs in rank_grouped_seqs.items():
        # Create SeqRecord objects for each sequence
        records = [SeqRecord(Seq(seq), id=f"{sk}_{i}") for i, seq in enumerate(seqs)]
        # Create a MultipleSeqAlignment object
        aln = MultipleSeqAlignment(records)
        # Compute the PSSM
        pssm = alignment_to_pssm(aln, freq=True)
        # Drop conservation column
        pssms_tax_dict[sk] = pssm.drop(columns=['cons_i'])

    # Example: show the PSSM for one superkingdom
    for sk, pssm in pssms_tax_dict.items():
        print(f"PSSM for {sk}:")
        display(pssm.head())
        break  # Remove this break to show all

    return pssms_tax_dict





## UPDATE: needs to be updated using the notebook that Frances sent
def build_potts_model(msa, pseudocount):
    """
    Build a naive Potts model from a multiple sequence alignment (MSA).
    Args:
        msa: list of strings, each string is a protein sequence of equal length.
    Returns:
      h   : single-site fields, shape (L, q)
      J   : pairwise couplings, shape (L, L, q, q)
    """

    # Map characters to integers
    chars = sorted(list(set("".join(msa))))
    c2i = {c: i for i, c in enumerate(chars)}
    N, L = len(msa), len(msa[0])
    q = len(chars)

    # Single frequencies
    fi = np.zeros((L, q))
    for seq in tqdm(msa, desc="Calculating single frequencies"):
        for i, c in enumerate(seq):
            fi[i, c2i[c]] += 1
    fi = (fi + pseudocount) / (N + q * pseudocount)

    # Pair frequencies
    fij = np.zeros((L, L, q, q))
    for seq in tqdm(msa, desc="Calculating pair frequencies"):
        for i in range(L):
            for j in range(i+1, L):
                ai, aj = c2i[seq[i]], c2i[seq[j]]
                fij[i,j,ai,aj] += 1
                fij[j,i,aj,ai] += 1
    fij = (fij + pseudocount) / (N + q * q * pseudocount)

    # Fields and couplings (naive approximation)
    h = -np.log(fi + 1e-12)
    J = np.zeros_like(fij)
    for i in tqdm(range(L), desc="Calculating fields and couplings"):
        for j in range(i+1, L):
            for a in range(q):
                for b in range(q):
                    if fi[i,a] > 0 and fi[j,b] > 0 and fij[i,j,a,b] > 0:
                        J[i,j,a,b] = -np.log(fij[i,j,a,b] / (fi[i,a]*fi[j,b]))
                        J[j,i,b,a] = J[i,j,a,b]
    return h, J