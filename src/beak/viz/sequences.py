"""Visualization functions for individual sequences and sequence comparisons"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from ..alignments.utils import SimilarityResult
except ImportError:
    # Handle case where utils module doesn't exist yet
    SimilarityResult = None


def plot_similar(similarity_result, figsize=None):
    """Plot similar sequences with shared residues highlighted in amber using matplotlib.

    Args:
        similarity_result (SimilarityResult): Result from find_similar() function
        figsize (tuple, optional): Figure size as (width, height) in inches.
                                 If None, will auto-calculate based on number of sequences.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
    """
    if SimilarityResult is None:
        raise ImportError("SimilarityResult not available. Make sure beak.alignments.utils is installed.")

    # Get sequence length and number of sequences
    seq_length = len(similarity_result.query_sequence)
    total_sequences = len(similarity_result.similar_sequences) + 1  # +1 for query

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(12, seq_length * 0.2)  # Scale width with sequence length
        height = max(6, total_sequences * 0.2)  # Scale height with number of sequences
        figsize = (width, height)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Define color mapping
    # 0 = black (non-matching), 1 = amber (matching), 2 = grey (masked out)
    colors = ['black', '#FFC000', '#D3D3D3']
    cmap = ListedColormap(colors)

    # Prepare data matrix and labels
    data_matrix = np.zeros((total_sequences, seq_length))
    sequence_labels = [None] * total_sequences  # Pre-allocate with correct size
    sequence_data = [None] * total_sequences    # Pre-allocate with correct size

    # Add query sequence (top row - will be at highest y index for top display)
    query_row = np.zeros(seq_length)
    for pos in range(seq_length):
        if similarity_result.masked_positions is not None:
            if pos in similarity_result.masked_positions:
                query_row[pos] = 1  # Amber for positions in mask (focused positions)
            else:
                query_row[pos] = 2  # Grey for positions not in mask
        else:
            query_row[pos] = 0  # Black if no masking

    query_row_idx = total_sequences - 1  # Top position
    data_matrix[query_row_idx] = query_row
    sequence_labels[query_row_idx] = f"Query: {similarity_result.query_id}"
    sequence_data[query_row_idx] = similarity_result.query_sequence

    # Add similar sequences (from bottom to top, excluding the query row)
    for seq_idx, (seq_id, sequence, similarity_score) in enumerate(similarity_result.similar_sequences):
        identity_vector = similarity_result.identity_vectors[seq_idx]
        seq_row = np.zeros(seq_length)

        for pos in range(seq_length):
            if similarity_result.masked_positions is not None and pos not in similarity_result.masked_positions:
                # Position is masked out - use grey
                seq_row[pos] = 2
            elif pos < len(identity_vector) and identity_vector[pos] == 1:
                # Matching residue - use amber
                seq_row[pos] = 1
            else:
                # Non-matching residue - use black
                seq_row[pos] = 0

        # Put similar sequences from bottom up (total_sequences - 2 - seq_idx)
        matrix_row_idx = total_sequences - 2 - seq_idx
        data_matrix[matrix_row_idx] = seq_row
        display_id = seq_id[:20] + "..." if len(seq_id) > 23 else seq_id
        sequence_labels[matrix_row_idx] = f"{display_id} ({similarity_score:.3f})"
        sequence_data[matrix_row_idx] = sequence

    # Create the heatmap
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

    # Add sequence letters as text overlay
    for seq_idx in range(total_sequences):
        sequence = sequence_data[seq_idx]
        for pos in range(seq_length):
            residue = sequence[pos] if pos < len(sequence) else '-'
            # Use white text for better contrast
            ax.text(pos, seq_idx, residue, ha='center', va='center',
                   color='white', fontsize=10, fontweight='bold')

    # Set up axes
    ax.set_xlim(-0.5, seq_length - 0.5)
    ax.set_ylim(-0.5, total_sequences - 0.5)

    # Set y-axis labels (sequence names)
    ax.set_yticks(range(total_sequences))
    ax.set_yticklabels(sequence_labels)

    # Set x-axis ticks (position numbers)
    x_ticks = np.arange(0, seq_length, max(1, seq_length // 20))  # Show ~20 ticks max
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i+1) for i in x_ticks])

    # Labels and title
    ax.set_xlabel('Sequence Position')
    ax.set_title('Sequence Similarity Comparison\n(Amber: matching residues, Grey: masked positions)')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig
