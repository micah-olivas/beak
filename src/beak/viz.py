import os
import urllib
import urllib.request

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from Bio.PDB import alphafold_db, PDBIO, PDBParser

import seqlogo
import logomaker

from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2

import panel as pn
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider

from .alignments.utils import SimilarityResult

from Bio.Data.IUPACData import protein_letters_3to1

def plot_seq_length_hist(aln, title=None, ref_length=None):
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
        # yticks=10,
        )
    return heatmap.opts(title=f"Aligned Position {position+1}")

def plot_conservation_with_sequence(query_record, pos, consensus, conservation, window=20):
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

def plot_similar(similarity_result: SimilarityResult, figsize=None):
    """Plot similar sequences with shared residues highlighted in amber using matplotlib.
    
    Args:
        similarity_result (SimilarityResult): Result from find_similar() function
        figsize (tuple, optional): Figure size as (width, height) in inches. 
                                 If None, will auto-calculate based on number of sequences.
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
    """
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

def plot_sequence_logo(PSSM, top_label=None, bottom_label=None, figsize=(12, 3), scroll=False):
    """
    Plot a sequence logo from a Position-Specific Scoring Matrix (PSSM).
    
    Parameters:
    -----------
    PSSM : pandas.DataFrame
        Position-Specific Scoring Matrix with positions as rows and amino acids as columns
    top_label : str, optional
        Label prefix for the y-axis (will be appended with "Enrichment")
    bottom_label : str, optional
        Currently unused parameter
    figsize : tuple
        Figure size (width, height)
    scroll : bool, optional
        If True, enables interactive scrolling along x-axis (50-residue window)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    logo : logomaker.Logo
        The logo object
    """

    # Create a copy and clean
    PSSM = PSSM.copy()
    for col in ['cons_i', '-']:
        if col in PSSM.columns:
            PSSM = PSSM.drop(col, axis=1)
    
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
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Labels
    if top_label is not None:
        ax.set_ylabel(f'{top_label}\nEnrichment', fontsize=12)
    ax.set_xlabel('Position', fontsize=12)
    y_min, y_max = ax.get_ylim()
    yticks = np.arange(np.floor(y_min / 0.2) * 0.2, np.ceil(y_max / 0.2) * 0.2 + 0.2, 0.2)
    ax.set_yticks(yticks)

    # Scroll functionality
    if scroll:
        window = 50
        max_pos = len(PSSM)
        ax.set_xlim(0, window)

        # Adjust layout for slider
        plt.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Position', 0, max_pos - window, valinit=0, valstep=1)

        def update(val):
            start = int(slider.val)
            ax.set_xlim(start, start + window)
            fig.canvas.draw_idle()

        slider.on_changed(update)

    plt.tight_layout()
    return fig, ax, logo

def three_to_one(resname):
    """Convert 3-letter residue name to 1-letter code (X if unknown)."""
    return protein_letters_3to1.get(resname.capitalize(), "X")

def save_mapped_structure(input, value_dict, reference_seq, null_bfactor=0.0):
    """
    Map desired position-wise values to the B-factor column of a PDB file,
    aligning the reference sequence to the PDB-derived sequence.

    Parameters
    ----------
    input : str
        Path to the input PDB file, PDB ID, or UniProt Accession Number.
    value_dict : dict
        Dict with keys as reference sequence positions (1-based) and values (float).
    reference_seq : str
        Reference amino acid sequence (1-letter code).
    null_bfactor : float, optional
        Default B-factor value for residues not in value_dict (default=0.0).

    Returns
    -------
    str
        Path to output PDB with mapped B-factors.
    """

    # --- Fetch structure if needed ---
    if ".pdb" not in input:
        if len(input) == 4:  # PDB ID
            pdb_id = input
            urllib.request.urlretrieve(
                f"http://files.rcsb.org/download/{pdb_id}.pdb", f"{pdb_id}.pdb"
            )
            pdb_filepath = f"{pdb_id}.pdb"
        elif os.path.isdir(input):
            print("dir! add support for directories...") # ADD DIR SUPPORT
            os.mkdir('mapped_structures')
        else:  # UniProt Accession → AlphaFold
            structures = alphafold_db.get_structural_models_for(input)
            io = PDBIO()
            for i, structure in enumerate(structures, start=1):
                pdb_filepath = f"{input}_AF_model_{i}.pdb"
                io.set_structure(structure)
                io.save(pdb_filepath)
                break
    else:
        pdb_filepath = input

    # --- Extract sequence from PDB ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_filepath)
    model = next(structure.get_models())
    chain = next(model.get_chains())  # assume single chain
    pdb_residues = [res for res in chain.get_residues() if res.id[0] == " "]
    pdb_seq = "".join(three_to_one(res.get_resname()) for res in pdb_residues)

    # --- Align reference sequence to PDB sequence ---
    alignment = pairwise2.align.globalxx(reference_seq, pdb_seq)[0]
    ref_aln, pdb_aln, score, start, end = alignment

    # --- Build mapping (ref_pos → pdb_residue) ---
    mapping = {}
    ref_pos, pdb_pos = 0, 0
    for r_char, p_char in zip(ref_aln, pdb_aln):
        if r_char != "-":
            ref_pos += 1
        if p_char != "-":
            pdb_pos += 1
        if r_char != "-" and p_char != "-":
            mapping[ref_pos] = pdb_residues[pdb_pos - 1]

    # --- Modify PDB file lines ---
    with open(pdb_filepath, "r") as pdb_file:
        pdb_lines = pdb_file.readlines()

    modified_pdb_lines = []
    for line in pdb_lines:
        if line.startswith("ATOM"):
            residue_index = int(line[22:26].strip())

            # set to null value first
            b_factor = null_bfactor

            # overwrite if mapping has a value
            for ref_pos, residue in mapping.items():
                if residue.get_id()[1] == residue_index:
                    if ref_pos in value_dict:
                        b_factor = value_dict[ref_pos]
                    break

            line = line[:60] + f"{b_factor:6.2f}" + line[66:]
            modified_pdb_lines.append(line)
        else:
            modified_pdb_lines.append(line)

    # --- Save modified file ---
    output_filepath = f"mapped_value_{os.path.basename(pdb_filepath)}"
    with open(output_filepath, "w") as f:
        f.writelines(modified_pdb_lines)

    print(f"Modified structure saved to {output_filepath}")
    return output_filepath