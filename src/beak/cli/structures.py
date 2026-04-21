"""Structure discovery, download, and per-residue feature extraction."""

import click
from pathlib import Path

from .main import main


@main.command()
@click.argument('uniprot_ids', nargs=-1)
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='File with UniProt IDs (one per line)')
@click.option('--source', type=click.Choice(['pdb', 'alphafold', 'both']),
              default='both', help='Structure source (default: both)')
@click.option('--selection', default='best',
              help='Selection strategy: "best", "all", or a number')
@click.option('--output-dir', '-o', default='structures',
              help='Output directory for structure files')
@click.option('--find-only', is_flag=True,
              help='Only discover structures, do not download')
def structures(uniprot_ids, input_file, source, selection, output_dir, find_only):
    """Find and download protein structures for UniProt IDs.

    Searches PDB (experimental) and AlphaFold (predicted) databases.

    UNIPROT_IDS are one or more UniProt accessions.

    Examples:

        beak structures P0DTC2 P00520

        beak structures -i uniprot_ids.txt --source pdb

        beak structures P0DTC2 --find-only

        beak structures P0DTC2 --selection all -o all_structures/
    """
    from .theme import get_console, BEAK_BLUE
    from rich.table import Table
    from rich.status import Status
    from ..api.structures import find_structures, fetch_structures as fetch_structs

    console = get_console()

    ids = list(uniprot_ids)
    if input_file:
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ids.append(line.split()[0])

    if not ids:
        raise click.UsageError(
            "Provide UniProt IDs as arguments or via --input file."
        )

    ids = list(dict.fromkeys(ids))

    console.print(f"[brand]Finding structures for {len(ids):,} UniProt IDs...[/brand]")

    status = Status("Querying databases...", console=console)
    status.start()

    def _on_find(processed, total):
        status.update(
            f"Querying PDBe SIFTS... [bold]{processed}/{total}[/bold] IDs"
        )

    try:
        avail_df = find_structures(ids, source=source, on_progress=_on_find)
    finally:
        status.stop()

    pdb_rows = avail_df[avail_df['source'] == 'pdb']
    af_rows = avail_df[avail_df['source'] == 'alphafold']

    console.print(
        f"  Found [bold]{len(pdb_rows)}[/bold] PDB entries, "
        f"[bold]{len(af_rows)}[/bold] AlphaFold models\n"
    )

    if avail_df.empty:
        console.print("[dim]No structures found.[/dim]")
        return

    summary_table = Table(title="Available Structures", border_style=BEAK_BLUE)
    summary_table.add_column("UniProt ID", style="bold")
    summary_table.add_column("Source")
    summary_table.add_column("Structure ID")
    summary_table.add_column("Chain")
    summary_table.add_column("Resolution", justify="right")
    summary_table.add_column("Method")

    display_limit = 30
    for _, row in avail_df.head(display_limit).iterrows():
        res_str = f"{row['resolution']:.2f}" if row['resolution'] else "-"
        summary_table.add_row(
            row['uniprot_id'],
            row['source'],
            row['structure_id'],
            row.get('chain_id', '-'),
            res_str,
            row.get('method', '')[:30],
        )

    console.print(summary_table)
    if len(avail_df) > display_limit:
        console.print(
            f"[dim]  ... and {len(avail_df) - display_limit} more[/dim]"
        )

    if find_only:
        return

    console.print()
    dl_status = Status("Downloading...", console=console)
    dl_status.start()

    def _on_dl(downloaded, total):
        dl_status.update(
            f"Downloading structures... [bold]{downloaded}/{total}[/bold]"
        )

    try:
        result_df = fetch_structs(
            avail_df,
            output_dir=output_dir,
            selection=selection,
            on_progress=_on_dl,
        )
    finally:
        dl_status.stop()

    downloaded = result_df['local_path'].notna().sum()
    failed = result_df['error'].notna().sum()
    console.print(
        f"\n[green]Downloaded {downloaded} structures to {output_dir}/[/green]"
    )
    if failed:
        console.print(f"[dim]{failed} unavailable (no structure found)[/dim]")


@main.command()
@click.argument('input_paths', nargs=-1, required=True)
@click.option('--chain', 'chain_id', default=None,
              help='Chain ID to extract (default: first polymer chain)')
@click.option('--output', '-o', 'output_path', default=None,
              help='Output file path (CSV or Parquet)')
@click.option('--format', 'output_format', default='csv',
              type=click.Choice(['csv', 'parquet']),
              help='Output format (default: csv)')
@click.option('--no-dssp', is_flag=True,
              help='Skip secondary structure assignment')
@click.option('--no-sasa', is_flag=True,
              help='Skip SASA computation (faster)')
def features(input_paths, chain_id, output_path, output_format, no_dssp, no_sasa):
    """Extract per-residue structural features from mmCIF files.

    INPUT_PATHS are mmCIF file paths or directories containing .cif files.

    Examples:

        beak features structures/P0DTC2_6VXX_A.cif

        beak features structures/ -o features.csv

        beak features structures/P0DTC2_AF.cif --chain A --no-dssp
    """
    from .theme import get_console, BEAK_BLUE
    from rich.table import Table
    from rich.status import Status
    from ..structures import extract_structure_features

    console = get_console()

    cif_files = []
    for path_str in input_paths:
        p = Path(path_str)
        if p.is_dir():
            cif_files.extend(sorted(p.glob('*.cif')))
        elif p.is_file() and p.suffix == '.cif':
            cif_files.append(p)
        else:
            raise click.BadParameter(
                f"'{path_str}' is not a .cif file or directory.",
                param_hint="'INPUT_PATHS'",
            )

    if not cif_files:
        console.print("[dim]No .cif files found.[/dim]")
        return

    console.print(
        f"[brand]Extracting features from {len(cif_files)} structure(s)...[/brand]"
    )

    all_dfs = []
    status = Status("Processing...", console=console)
    status.start()

    try:
        for i, cif_path in enumerate(cif_files, 1):
            status.update(
                f"Processing [bold]{cif_path.name}[/bold] "
                f"[dim]({i}/{len(cif_files)})[/dim]"
            )
            try:
                df = extract_structure_features(
                    str(cif_path),
                    chain_id=chain_id,
                    skip_dssp=no_dssp,
                    skip_sasa=no_sasa,
                )
                df.insert(0, 'structure', cif_path.stem)
                all_dfs.append(df)
            except Exception as e:
                console.print(f"  [red]Error processing {cif_path.name}: {e}[/red]")
    finally:
        status.stop()

    if not all_dfs:
        console.print("[dim]No features extracted.[/dim]")
        return

    import pandas as pd
    result = pd.concat(all_dfs, ignore_index=True)

    if output_path:
        if output_format == 'parquet':
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
        console.print(
            f"[green]Wrote {len(result)} rows to {output_path}[/green]"
        )
    else:
        table = Table(
            title=f"Structural Features ({len(result)} residues)",
            border_style=BEAK_BLUE,
        )
        table.add_column("Structure", style="dim")
        table.add_column("Res#", justify="right")
        table.add_column("AA", style="bold")
        table.add_column("pLDDT", justify="right")
        table.add_column("SS")
        table.add_column("SASA", justify="right")
        table.add_column("Contacts", justify="right")
        table.add_column("WCN", justify="right")

        display_limit = 50
        for _, row in result.head(display_limit).iterrows():
            plddt_str = f"{row['plddt']:.1f}" if pd.notna(row['plddt']) else "-"
            sasa_str = f"{row['sasa']:.1f}" if pd.notna(row['sasa']) else "-"
            wcn_str = f"{row['burial_wcn']:.3f}" if pd.notna(row['burial_wcn']) else "-"

            table.add_row(
                row['structure'],
                str(row['residue_number']),
                row['residue_name'],
                plddt_str,
                row.get('secondary_structure', '-'),
                sasa_str,
                str(row['n_contacts']),
                wcn_str,
            )

        console.print(table)
        if len(result) > display_limit:
            console.print(
                f"[dim]  ... and {len(result) - display_limit} more residues "
                f"(use -o to save full output)[/dim]"
            )
