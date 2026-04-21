"""`beak pfam` command — Pfam domain scanning plus optional UniProt lookup."""

import click

from .main import main
from ._common import get_hmmer_manager, resolve_query_to_fasta


@main.command()
@click.argument('query', required=False)
@click.option('--uniprot', is_flag=True,
              help='Fetch UniProt IDs for discovered domains')
@click.option('--pfam', 'pfam_id', default=None,
              help='Skip scan; look up a known Pfam ID directly (e.g., PF00069)')
@click.option('--taxonomy', is_flag=True,
              help='Include organism name and tax ID in UniProt results')
@click.option('--lineage', is_flag=True,
              help='Include full taxonomic lineage (implies --taxonomy)')
@click.option('--evalue', default=1e-5, type=float,
              help='E-value threshold for hmmscan (default: 1e-5)')
@click.option('--structures', 'fetch_structures_flag', is_flag=True,
              help='Download structures for discovered UniProt IDs')
@click.option('--output-dir', '-o', default='structures',
              help='Output directory for structures (with --structures)')
def pfam(query, uniprot, pfam_id, taxonomy, lineage, evalue,
         fetch_structures_flag, output_dir):
    """Scan a protein sequence for Pfam domains.

    QUERY can be a FASTA file, a UniProt accession (e.g., P0DTC2), or a raw
    amino acid sequence string.

    With --uniprot, also fetches UniProt proteins sharing those domains.
    With --pfam PF00069, skips the scan and goes straight to UniProt lookup.
    With --structures, downloads PDB/AlphaFold structures for UniProt hits.

    Examples:

        beak pfam my_seq.fasta

        beak pfam P0DTC2 --uniprot --taxonomy

        beak pfam MKLFVLARLC... --uniprot

        beak pfam --pfam PF00069 --uniprot --lineage

        beak pfam --pfam PF00069 --uniprot --structures -o kinase_structures/
    """
    from .theme import get_console, BEAK_BLUE
    from rich.table import Table

    console = get_console()

    if lineage:
        taxonomy = True

    if pfam_id:
        domains = [{'pfam_id': pfam_id, 'pfam_name': pfam_id,
                    'description': '', 'i_evalue': '-'}]
        if not uniprot:
            uniprot = True
    else:
        if not query:
            raise click.UsageError(
                "QUERY is required unless --pfam is provided."
            )

        query_file = resolve_query_to_fasta(query)
        console.print(f"[brand]Scanning for Pfam domains...[/brand]")
        _, hmmer = get_hmmer_manager()
        domains = hmmer.scan(query_file, evalue=evalue)

        if not domains:
            console.print("[dim]No Pfam domains found.[/dim]")
            return

        table = Table(title="Pfam Domain Hits", border_style=BEAK_BLUE)
        table.add_column("Pfam ID", style="bold")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("E-value", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Region", justify="right")

        for hit in domains:
            table.add_row(
                hit['pfam_id'],
                hit['pfam_name'],
                hit['description'][:50],
                f"{hit['i_evalue']:.1e}",
                f"{hit['score']:.1f}",
                f"{hit['ali_from']}-{hit['ali_to']}",
            )

        console.print(table)

    if uniprot:
        from ..api.uniprot import query_uniprot_by_pfam

        for domain in domains:
            pid = domain['pfam_id']
            from rich.status import Status

            console.print(f"\n[brand]Fetching UniProt entries for {pid}...[/brand]")

            status = Status("Querying UniProt...", console=console)
            status.start()

            def _on_progress(count, page):
                status.update(
                    f"Fetched [bold]{count:,}[/bold] entries "
                    f"[dim](page {page})[/dim]"
                )

            try:
                df = query_uniprot_by_pfam(
                    pid,
                    taxonomy=taxonomy,
                    lineage=lineage,
                    on_progress=_on_progress,
                )
            finally:
                status.stop()

            if df.empty:
                console.print(f"[dim]No UniProt entries found for {pid}.[/dim]")
                continue

            header = f"{pid} — {len(df)} UniProt entries"
            result_table = Table(title=header, border_style=BEAK_BLUE)
            result_table.add_column("Accession", style="bold")
            result_table.add_column("Protein Name")

            if taxonomy:
                result_table.add_column("Organism")
                result_table.add_column("Tax ID", justify="right")
            if lineage:
                result_table.add_column("Lineage")

            display_limit = 50
            for _, row in df.head(display_limit).iterrows():
                cols = [row['accession'], row['protein_name'][:60]]
                if taxonomy:
                    cols.append(row.get('organism_name', ''))
                    cols.append(str(row.get('organism_id', '')))
                if lineage:
                    cols.append(row.get('lineage', '')[:80])
                result_table.add_row(*cols)

            console.print(result_table)
            if len(df) > display_limit:
                console.print(
                    f"[dim]  ... and {len(df) - display_limit} more entries "
                    f"(use Python API for full results)[/dim]"
                )

    if fetch_structures_flag and uniprot:
        from ..api.structures import find_structures, fetch_structures as fetch_structs
        from ..api.uniprot import query_uniprot_by_pfam
        from rich.status import Status as StructStatus

        all_accessions = []
        for domain in domains:
            pid = domain['pfam_id']
            domain_df = query_uniprot_by_pfam(pid, taxonomy=False)
            all_accessions.extend(domain_df['accession'].tolist())

        if all_accessions:
            all_accessions = list(dict.fromkeys(all_accessions))

            console.print(
                f"\n[brand]Finding structures for "
                f"{len(all_accessions):,} UniProt IDs...[/brand]"
            )

            struct_status = StructStatus("Querying PDBe SIFTS...", console=console)
            struct_status.start()

            def _on_find_progress(processed, total):
                struct_status.update(
                    f"Querying PDBe SIFTS... "
                    f"[bold]{processed}/{total}[/bold] IDs"
                )

            try:
                avail_df = find_structures(
                    all_accessions, on_progress=_on_find_progress,
                )
            finally:
                struct_status.stop()

            pdb_count = len(avail_df[avail_df['source'] == 'pdb'])
            af_count = len(avail_df[avail_df['source'] == 'alphafold'])
            console.print(
                f"  Found [bold]{pdb_count}[/bold] PDB entries, "
                f"[bold]{af_count}[/bold] AlphaFold models"
            )

            dl_status = StructStatus("Downloading...", console=console)
            dl_status.start()

            def _on_dl_progress(downloaded, total):
                dl_status.update(
                    f"Downloading structures... "
                    f"[bold]{downloaded}/{total}[/bold]"
                )

            try:
                result_df = fetch_structs(
                    avail_df,
                    output_dir=output_dir,
                    selection="best",
                    on_progress=_on_dl_progress,
                )
            finally:
                dl_status.stop()

            downloaded = result_df['local_path'].notna().sum()
            failed = result_df['error'].notna().sum()
            console.print(
                f"  Downloaded [bold]{downloaded}[/bold] structures "
                f"to [cyan]{output_dir}/[/cyan]"
            )
            if failed:
                console.print(
                    f"  [dim]{failed} unavailable (no structure)[/dim]"
                )
