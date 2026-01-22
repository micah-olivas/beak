"""Visualization functions for multiple sequence alignments"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Suppress deprecation warnings from third-party libraries
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='.*Bio.pairwise2 has been deprecated.*')

import seqlogo
import logomaker
import panel as pn
import holoviews as hv


def plot_seq_length_hist(aln, title=None, ref_length=None):
    """
    Plot histogram of sequence lengths in an alignment.

    Args:
        aln: BioPython MultipleSeqAlignment object
        title: Optional title for the plot
        ref_length: Optional reference length to mark with vertical line
    """
    seq_lengths = [len(str(record.seq).replace('.', '').replace('-', '')) for record in aln]
    plt.figure(figsize=(4,3))
    plt.hist(seq_lengths, bins=30, color='skyblue', edgecolor='black')

    if ref_length != None:
        plt.axvline(ref_length, color='red', linestyle='dashed', linewidth=2, label=f'Aligned Length: {ref_length}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Sequence Length')
    plt.ylabel('Count')

    # Add N sequences in top left
    plt.text(0.05, 0.95, f'N: {len(seq_lengths)}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    if title:
        plt.title(title)

    plt.show()


def plot_sequence_logo_seqlogo(pssm, title="Sequence Logo"):
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


def plot_sequence_logo(PSSM, top_label=None, bottom_label=None, figsize=None, interactive=False,
                       motif=None, consensus=None, selected_ref_positions=None, alignment=None):
    """
    Plot a sequence logo from a Position-Specific Scoring Matrix (PSSM).

    Parameters:
    -----------
    PSSM : pandas.DataFrame
        Position-Specific Scoring Matrix with positions as rows and amino acids as columns
    top_label : str, optional
        Label prefix for the y-axis
    bottom_label : str, optional
        Currently unused parameter
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated based on PSSM/motif length
    interactive : bool, optional
        If True, enables interactive pan/zoom capabilities. Requires ipympl and
        %matplotlib widget in Jupyter notebooks. Default is False.
    motif : str, optional
        Regex pattern to search for (e.g., "G.VQGV"). If found, only those positions are shown.
    consensus : str, optional
        Consensus sequence to search for motif. If None and motif is provided, uses PSSM to generate one.
    selected_ref_positions : dict, optional
        Dictionary mapping reference sequence ID to list of ungapped positions.
        E.g., {"UniRef90_P14621": [10, 15, 20]} shows positions 10, 15, 20 from that reference.
        Requires alignment parameter to be provided.
    alignment : MultipleSeqAlignment, optional
        Required when using selected_ref_positions to map ungapped to gapped positions.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    logo : logomaker.Logo
        The logo object

    Examples:
    ---------
    Basic usage:
        >>> fig, ax, logo = plot_sequence_logo(PSSM)

    Interactive mode (in Jupyter notebook):
        >>> %matplotlib widget
        >>> fig, ax, logo = plot_sequence_logo(PSSM, interactive=True)
        # Now use the toolbar to pan and zoom!

    With motif filtering:
        >>> fig, ax, logo = plot_sequence_logo(PSSM, motif="G.VQGV", interactive=True)
    """
    # Create a copy and clean
    PSSM = PSSM.copy()

    # Remove X column to avoid logomaker warning
    if 'X' in PSSM.columns:
        PSSM = PSSM.drop('X', axis=1)

    for col in ['cons_i', '-']:
        if col in PSSM.columns:
            PSSM = PSSM.drop(col, axis=1)

    # Detect if PSSM is frequencies or counts
    is_frequency = PSSM.max().max() <= 1.0

    # Handle selected_ref_positions
    ref_gapped_positions = None
    ref_ungapped_positions = None
    ref_id = None
    ref_sequence = None
    if selected_ref_positions is not None:
        if alignment is None:
            raise ValueError("alignment parameter is required when using selected_ref_positions")

        # Get reference ID and positions
        if len(selected_ref_positions) != 1:
            raise ValueError("selected_ref_positions must contain exactly one reference sequence")

        ref_id, ref_ungapped_positions = list(selected_ref_positions.items())[0]

        # Find reference sequence in alignment
        ref_record = None
        for record in alignment:
            if record.id == ref_id:
                ref_record = record
                break

        if ref_record is None:
            raise ValueError(f"Reference sequence '{ref_id}' not found in alignment")

        ref_sequence = str(ref_record.seq)

        # Map ungapped positions to gapped (alignment) positions
        ref_gapped_positions = []
        ungapped_pos = 0
        for gapped_pos, char in enumerate(ref_sequence):
            if char != '-':
                # This is an ungapped position
                if ungapped_pos in ref_ungapped_positions:
                    ref_gapped_positions.append(gapped_pos)
                ungapped_pos += 1

        if len(ref_gapped_positions) == 0:
            raise ValueError(f"None of the specified positions found in reference sequence")

        # Filter PSSM to only show selected positions
        PSSM = PSSM.iloc[ref_gapped_positions].copy()
        PSSM.index = range(len(PSSM))

    # Handle motif filtering
    motif_positions = None
    motif_match = None
    motif_query = None
    if motif is not None and selected_ref_positions is None:
        motif_query = motif  # Store original query

        # Generate consensus if not provided
        if consensus is None:
            consensus = ''.join([PSSM.iloc[i].idxmax() for i in range(len(PSSM))])

        # Search for motif
        match = re.search(motif, consensus)
        if match:
            motif_match = match.group()
            start_pos = match.start()
            end_pos = match.end()
            motif_positions = list(range(start_pos, end_pos))

            # Filter PSSM to only motif positions
            PSSM = PSSM.iloc[motif_positions].copy()
            # Reset index for clean plotting
            PSSM.index = range(len(PSSM))
        else:
            print(f"Warning: Motif '{motif}' not found in consensus sequence")
            motif_positions = None

    # Auto-calculate figure size if not provided
    if figsize is None:
        n_positions = len(PSSM)
        # For interactive mode, use wider default to show more positions
        if interactive:
            width = max(12, min(20, n_positions * 0.3))
        else:
            width = max(6, n_positions * 0.5)
        height = 4  # Taller for better visibility
        figsize = (width, height)

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

    # Create the logo
    logo = logomaker.Logo(
        PSSM,
        ax=ax,
        color_scheme='chemistry',
        font_name='Arial'
    )

    # Style
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Dynamic y-label based on data type
    ylabel = 'Frequency' if is_frequency else 'Count'
    if top_label is not None:
        ylabel = f'{top_label}\n{ylabel}'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Position', fontsize=12)

    y_min, y_max = ax.get_ylim()
    yticks = np.arange(np.floor(y_min / 0.2) * 0.2, np.ceil(y_max / 0.2) * 0.2 + 0.2, 0.2)
    ax.set_yticks(yticks)

    # Add reference position annotations if using selected_ref_positions
    if ref_gapped_positions is not None:
        # Add reference ID at top
        ax.text(0.5, 1.25, f'Reference: {ref_id}',
               transform=ax.transAxes,
               ha='center',
               fontsize=14,
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Add alignment position indices (gray, small)
        for i, gapped_pos in enumerate(ref_gapped_positions):
            ax.text(i, 1.05, str(gapped_pos + 1),  # 1-indexed alignment position
                   ha='center',
                   va='bottom',
                   fontsize=7,
                   color='gray')

        # Add ungapped reference positions (bold, larger)
        for i, ungapped_pos in enumerate(ref_ungapped_positions):
            ax.text(i, 1.1, str(ungapped_pos),  # Ungapped position as specified
                   ha='center',
                   va='bottom',
                   fontsize=9,
                   fontweight='bold',
                   color='black')

        # Add reference amino acids
        ref_chars = [ref_sequence[gapped_pos] for gapped_pos in ref_gapped_positions]
        for i, char in enumerate(ref_chars):
            ax.text(i, 1.15, char,
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   fontweight='bold',
                   color='darkblue')

    # Add motif annotation if found (and not using selected_ref_positions)
    elif motif_positions is not None and motif_match is not None:
        # Add motif query at top (not the matched sequence)
        ax.text(0.5, 1.15, f'Motif: {motif_query}',
               transform=ax.transAxes,
               ha='center',
               fontsize=14,
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add original position indices above each position
        original_positions = [motif_positions[i] + 1 for i in range(len(motif_positions))]  # 1-indexed
        for i, orig_pos in enumerate(original_positions):
            ax.text(i, 1.05, str(orig_pos),
                   ha='center',
                   va='bottom',
                   fontsize=8,
                   color='gray')

        # Add motif characters above position indices
        for i, char in enumerate(motif_match):
            ax.text(i, 1.1, char,
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   fontweight='bold',
                   color='black')

    # Enable interactive features if requested
    if interactive:
        # Check if ipympl backend is available
        import matplotlib
        current_backend = matplotlib.get_backend()

        if current_backend.lower() not in ['widget', 'nbagg', 'webagg']:
            print("=" * 70)
            print("⚠️  Interactive mode requested but matplotlib backend is not interactive")
            print("=" * 70)
            print("To enable interactive pan/zoom in Jupyter notebooks, run:")
            print("    %matplotlib widget")
            print()
            print("Then re-run this plot command.")
            print("=" * 70)
        else:
            print("✓ Interactive mode enabled - use the toolbar to pan and zoom!")
            print(f"  Showing {len(PSSM)} positions total")

    plt.tight_layout()
    return fig, ax, logo


def plot_pssm_single_site_taxa_heatmap(pssms_by_tax_rank, position, aas, cmap, rank):
    """
    Plot a heatmap of PSSM frequencies for a single site across different taxa.

    Args:
        pssms_by_tax_rank (dict): Dictionary of PSSMs indexed by taxonomic rank.
        position (int): Position in the PSSM to plot.
        aas (list): List of amino acids to include in the heatmap.
        cmap (str): Colormap for the heatmap.
        rank (str): Taxonomic rank for the y-axis label.

    Returns:
        hv.HeatMap: Heatmap of PSSM frequencies.
    """
    # Build DataFrame for the selected position
    heatmap_data = []
    index = sorted(pssms_by_tax_rank.keys(), reverse=True)  # Sort taxa alphabetically (A at the top)
    for sk in index:
        pssm = pssms_by_tax_rank[sk]
        row = []
        for aa in aas:
            row.append(pssm.iloc[position][aa] if aa in pssm.columns else 0)
        heatmap_data.append(row)

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, columns=aas, index=index)
    tidy = heatmap_df.reset_index().melt(id_vars='index', var_name='AA', value_name='Frequency')
    heatmap = hv.HeatMap(tidy, kdims=['AA', 'index'], vdims='Frequency').opts(
        cmap=cmap,
        colorbar=True,
        clim=(0, 1),
        xrotation=0,
        yrotation=0,
        xlabel='Amino Acid',
        ylabel=rank.capitalize(),
        colorbar_opts={'title': 'Frequency'},
        tools=['hover'],
        width=600,
        height=max(100, 30 * len(index)),
        line_color='black',
        show_grid=True,
        toolbar='above',
        labelled=['x', 'y', 'colorbar'],
        xaxis='top',
        fontsize={'xticks': 14, 'yticks': 10, 'ylabel': 14, 'xlabel': 14, 'title': 16},
        )
    return heatmap.opts(title=f"Aligned Position {position+1}")


def plot_conservation_with_sequence(query_record, pos, consensus, conservation, window=20):
    """
    Plot conservation scores with query and consensus sequences in a window.

    Args:
        query_record: BioPython SeqRecord of the query sequence
        pos: Position to highlight
        consensus: Consensus sequence string
        conservation: Array of conservation scores
        window: Size of window to display around position

    Returns:
        matplotlib.figure.Figure
    """
    half = window // 2
    start = max(0, pos - half)
    end = min(len(query_record.seq), start + window)
    start = max(0, end - window)
    window_seq = query_record.seq[start:end]

    fig, axs = plt.subplots(3, 1, figsize=(3, 1.5), gridspec_kw={'height_ratios': [1, 1, 1.5]}, constrained_layout=True, dpi=300)
    fig.patch.set_facecolor('white')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].set_facecolor('white')

    # Font
    fontdict = {'family': 'Helvetica Neue', 'weight': 'bold'}

    # Query sequence row
    for i, aa in enumerate(window_seq, start):
        color = 'orange' if i == pos else 'black'
        size = 12 if i == pos else 8
        weight = 'bold' if i == pos else 'normal'
        axs[0].text(i - start, 0.5, aa, ha='center', va='center', fontsize=size, color=color, fontdict={**fontdict, 'weight': weight})
    axs[0].set_xlim(-0.5, len(window_seq) - 0.5)
    axs[0].set_ylim(0, 1)
    axs[0].set_title(f"Query ({query_record.id})", loc='left', fontsize=8, fontdict=fontdict, y=0.6, color='grey')
    axs[0].set_xticks(np.arange(len(window_seq)))
    axs[0].set_xticklabels([])
    axs[0].tick_params(axis='x', which='both', length=3, labelbottom=False)

    # Consensus sequence row
    consensus_window = consensus[start:end]
    for i, aa in enumerate(consensus_window, start):
        color = 'orange' if i == pos else 'gray'
        size = 12 if i == pos else 8
        weight = 'bold' if i == pos else 'normal'
        axs[1].text(i - start, 0.5, aa, ha='center', va='center', fontsize=size, color=color, fontdict={**fontdict, 'weight': weight})
    axs[1].set_xlim(-0.5, len(consensus_window) - 0.5)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Consensus", loc='left', fontsize=8, fontdict=fontdict, y=0.62, color='grey')

    # Add aligned ticks (no labels)
    axs[1].set_xticks(np.arange(len(consensus_window)))
    axs[1].set_xticklabels([])
    axs[1].tick_params(axis='x', which='both', length=3, labelbottom=False)

    # Conservation bar plot (bottom row)
    conservation_window = conservation[start:end]
    bar_colors = ['orange' if i == pos - start else 'lightgray' for i in range(len(conservation_window))]
    axs[2].bar(
        np.arange(len(conservation_window)),
        conservation_window,
        color=bar_colors,
        width=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    axs[2].set_xlim(-0.5, len(conservation_window) - 0.5)
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Conservation', fontsize=7, fontstyle='italic')
    axs[2].set_xlabel('Aligned Position', fontsize=7, fontstyle='italic')

    # Major ticks every 5, minor ticks every 1
    xticks = np.arange(0, len(conservation_window), 5)
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels([str(start + i + 1) for i in xticks], fontsize=7)
    axs[2].set_yticks([0, 0.5, 1])
    axs[2].tick_params(axis='y', labelsize=7)

    # Add minor ticks
    axs[2].set_xticks(np.arange(0, len(conservation_window)), minor=True)
    axs[2].tick_params(axis='x', which='minor', length=2)
    axs[2].tick_params(axis='x', which='major', length=5)
    # Only show left and bottom spines, decrease line width
    for spine_name, spine in axs[2].spines.items():
        if spine_name in ['left', 'bottom']:
            spine.set_visible(True)
            spine.set_linewidth(1)
        else:
            spine.set_visible(False)

    return fig


def interactive_pssm_heatmap(
    pssms_by_tax_rank,
    consensus,
    rank,
    query_record,
    conservation,
):
    """
    Interactive PSSM heatmap explorer by taxonomic rank.

    Args:
        pssms_by_tax_rank: dict of {taxon: PSSM DataFrame}
        consensus: consensus sequence string
        rank: taxonomic rank label for y-axis
        query_record: BioPython SeqRecord for query
        conservation: conservation scores array

    Returns:
        pn.Column Panel layout for interactive exploration
    """
    # Prepare amino acid set
    aas = set()
    for pssm in pssms_by_tax_rank.values():
        aas.update(pssm.columns)
    aas = sorted(aas)

    # Custom colormap: 0 is white, 1 is blue
    cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    slider = pn.widgets.DiscreteSlider(
        name='Aligned Position',
        options=list(range(len(consensus))),
        value=0
    )

    @pn.depends(slider)
    def query_seq_matplotlib_view(position):
        return pn.pane.Matplotlib(plot_conservation_with_sequence(query_record, position, consensus, conservation, window=20))

    dmap = hv.DynamicMap(pn.bind(plot_pssm_single_site_taxa_heatmap, pssms_by_tax_rank=pssms_by_tax_rank, position=slider, aas=aas, cmap=cmap, rank=rank))

    return pn.Column(
        slider,
        pn.Row(
            dmap,
            pn.Column(
                query_seq_matplotlib_view,
                align='center'
            )
        )
    )
