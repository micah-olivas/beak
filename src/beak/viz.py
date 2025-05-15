import seqlogo
import numpy as np
import pandas as pd

import panel as pn
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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