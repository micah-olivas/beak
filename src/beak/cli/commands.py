"""BEAK command-line interface.

Usage:
    beak config init          # Set up connection defaults
    beak config show          # Show current config
    beak config set KEY VAL   # Set a config value

    beak jobs                 # List all jobs
    beak status JOB_ID        # Check job status
    beak log JOB_ID           # View job log
    beak cancel JOB_ID        # Cancel a running job
    beak results JOB_ID       # Download results

    beak search QUERY --db DB # Submit search job
    beak taxonomy QUERY       # Submit taxonomy job
    beak align INPUT          # Submit alignment job
    beak pfam QUERY           # Scan for Pfam domains

    beak setup pfam           # Set up Pfam database on remote
    beak databases            # List available databases
    beak doctor               # Check remote environment
"""

import json
import click
from pathlib import Path


def _get_manager(job_id=None, job_type=None):
    """Create the appropriate manager for a job_id or job_type.

    Reads job_type from the local job database when job_id is given.
    Uses config defaults for connection parameters.
    """
    from ..config import get_default_connection

    defaults = get_default_connection()
    host = defaults.get('host')
    user = defaults.get('user')

    if not host or not user:
        raise click.ClickException(
            "No connection configured. Run 'beak config init' first."
        )

    if job_id and not job_type:
        db_path = Path.home() / ".beak" / "jobs.json"
        if db_path.exists():
            with open(db_path) as f:
                job_db = json.load(f)
            if job_id in job_db:
                job_type = job_db[job_id].get('job_type')

    type_map = {
        'search': 'beak.remote.search:MMseqsSearch',
        'taxonomy': 'beak.remote.taxonomy:MMseqsTaxonomy',
        'align': 'beak.remote.align:ClustalAlign',
        'embeddings': 'beak.remote.embeddings:ESMEmbeddings',
        'pipeline': 'beak.remote.pipeline:Pipeline',
    }

    module_class = type_map.get(job_type, 'beak.remote.base:RemoteJobManager')
    module_path, class_name = module_class.split(':')

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    try:
        return cls()
    except (ConnectionError, TimeoutError, OSError) as e:
        raise click.ClickException(f"Cannot connect to remote server: {e}")


# ── Main CLI group ──────────────────────────────────────────────

@click.group()
@click.version_option(package_name='beak')
def main():
    """BEAK - Biophysical and Evolutionary Analysis Kit"""
    pass


# ── Doctor command ──────────────────────────────────────────────

@main.command()
def doctor():
    """Check remote server for required tools and databases"""
    from .theme import get_console, BEAK_BLUE
    from rich.table import Table
    from rich.text import Text

    console = get_console()
    mgr = _get_manager(job_type='search')
    results = mgr.verify_remote(verbose=False)

    console.print(f"\n[brand]BEAK Doctor[/brand]")
    console.print(f"[dim]Remote: {mgr.conn.host}[/dim]\n")

    # ── Tools table ──
    tools_table = Table(title="Tools", border_style=BEAK_BLUE, show_lines=False)
    tools_table.add_column("Tool", style="bold")
    tools_table.add_column("Status")
    tools_table.add_column("Version", style="dim")
    tools_table.add_column("Needed By", style="dim")

    for tool, info in results['tools'].items():
        if info['found']:
            status = "[green]OK[/green]"
            version = info.get('version', '') or ''
        elif info['required']:
            status = "[red]MISSING[/red]"
            version = info.get('install', '')
        else:
            status = "[dim]--[/dim]"
            version = ''
        tools_table.add_row(tool, status, version[:50], info['needed_by'])

    console.print(tools_table)
    console.print()

    # ── Databases table ──
    db_table = Table(title="Databases", border_style=BEAK_BLUE, show_lines=False)
    db_table.add_column("Database", style="bold")
    db_table.add_column("Status")
    db_table.add_column("Age")

    # MMseqs2 databases
    db_info = results.get('databases', {})
    if db_info.get('exists'):
        db_count = db_info.get('count', 0)
        db_table.add_row(
            f"MMseqs2 ({db_count} dbs)",
            "[green]OK[/green]",
            f"[dim]{db_info['path']}[/dim]",
        )
    else:
        db_table.add_row(
            "MMseqs2",
            "[red]MISSING[/red]",
            "[dim]directory not found[/dim]",
        )

    # Pfam database
    from ..remote.hmmer import resolve_pfam_path
    try:
        pfam_path = resolve_pfam_path(mgr.conn)
        from ..remote.hmmer import PFAM_HMM_FILE
        age_str = _get_remote_file_age(mgr.conn, f"{pfam_path}/{PFAM_HMM_FILE}")
        db_table.add_row("Pfam-A", "[green]OK[/green]", age_str)
    except FileNotFoundError:
        db_table.add_row(
            "Pfam-A",
            "[dim]--[/dim]",
            "[dim]not installed (beak setup pfam)[/dim]",
        )

    console.print(db_table)

    # ── Disk space ──
    disk = results.get('disk', {})
    if disk:
        console.print(
            f"\n[dim]Disk: {disk.get('available', '?')} available "
            f"of {disk.get('total', '?')} total[/dim]"
        )

    # ── Summary ──
    console.print()
    if results['ok']:
        console.print("[green]All required tools found.[/green]")
    else:
        missing = [t for t, s in results['tools'].items()
                   if s['required'] and not s['found']]
        if missing:
            console.print(f"[red]Missing required tools: {', '.join(missing)}[/red]")
            console.print("[dim]Install them before submitting jobs.[/dim]")
    console.print()


# ── Config commands ─────────────────────────────────────────────

@main.group()
def config():
    """Manage beak configuration"""
    pass


@config.command('init')
def config_init():
    """Interactive setup of ~/.beak/config.toml"""
    from ..config import save_config, CONFIG_PATH

    click.echo("BEAK Configuration Setup")
    click.echo("=" * 40)

    host = click.prompt("Remote server hostname", type=str)
    user = click.prompt("SSH username", type=str)
    key_path = click.prompt(
        "SSH key path",
        default="~/.ssh/id_ed25519",
        type=str
    )
    remote_job_dir = click.prompt(
        "Remote job directory",
        default="~/beak_jobs",
        type=str
    )

    config_data = {
        'connection': {
            'host': host,
            'user': user,
            'key_path': key_path,
            'remote_job_dir': remote_job_dir,
        },
        'projects': {
            'local_dir': '~/beak_projects',
        }
    }

    save_config(config_data)
    click.echo(f"\n✓ Configuration saved to {CONFIG_PATH}")


@config.command('show')
def config_show():
    """Show current configuration"""
    from ..config import load_config, CONFIG_PATH

    if not CONFIG_PATH.exists():
        click.echo("No configuration found. Run 'beak config init' to set up.")
        return

    config_data = load_config()

    for section, values in config_data.items():
        click.echo(f"[{section}]")
        if isinstance(values, dict):
            for key, val in values.items():
                click.echo(f"  {key} = {val}")
        else:
            click.echo(f"  {values}")
        click.echo()


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set a configuration value (e.g., beak config set connection.host myserver)"""
    from ..config import set_config_value

    set_config_value(key, value)
    click.echo(f"✓ Set {key} = {value}")


# ── Job management commands ─────────────────────────────────────

@main.command()
@click.option('--type', 'job_type', default=None,
              type=click.Choice(['search', 'taxonomy', 'align', 'embeddings', 'pipeline']),
              help='Filter by job type')
@click.option('--status', 'status_filter', default=None,
              help='Filter by status (RUNNING, COMPLETED, FAILED, etc.)')
@click.option('--refresh', is_flag=True,
              help='Connect to remote server to refresh job statuses')
def jobs(job_type, status_filter, refresh):
    """List all jobs"""
    db_path = Path.home() / ".beak" / "jobs.json"
    if not db_path.exists():
        click.echo("No jobs found.")
        return

    with open(db_path) as f:
        job_db = json.load(f)

    if not job_db:
        click.echo("No jobs found.")
        return

    # Refresh statuses from remote if requested
    if refresh:
        try:
            mgr = _get_manager(job_type='search')  # any manager works for status()
            updated = 0
            for job_id, info in job_db.items():
                old = info.get('status', 'UNKNOWN')
                if old in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    continue  # terminal states don't change
                result = mgr.status(job_id)
                if result['status'] != old:
                    updated += 1
            # Reload after status() calls updated the DB
            with open(db_path) as f:
                job_db = json.load(f)
            if updated:
                click.echo(f"Refreshed {updated} job(s)\n")
        except Exception as e:
            click.echo(f"Could not refresh: {e}\n")

    rows = []
    for job_id, info in job_db.items():
        jtype = info.get('job_type', 'unknown')
        if job_type and jtype != job_type:
            continue

        jstatus = info.get('status', 'UNKNOWN')
        if status_filter and jstatus != status_filter:
            continue

        rows.append({
            'id': job_id,
            'name': info.get('name', ''),
            'type': jtype,
            'status': jstatus,
            'submitted': info.get('submitted_at', '')[:19],
        })

    if not rows:
        click.echo("No matching jobs.")
        return

    # Truncate long names to fit the column
    name_width = 20
    header = f"{'ID':<10} {'Name':<{name_width}} {'Type':<12} {'Status':<12} {'Submitted'}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        name = r['name']
        if len(name) > name_width:
            name = name[:name_width - 1] + '…'
        click.echo(f"{r['id']:<10} {name:<{name_width}} {r['type']:<12} {r['status']:<12} {r['submitted']}")


@main.command()
@click.argument('job_id')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress')
@click.option('--watch', '-w', is_flag=True, help='Live-updating status display')
@click.option('--interval', default=2.0, help='Refresh interval in seconds (with --watch)')
def status(job_id, verbose, watch, interval):
    """Check job status"""
    from .display import print_status, watch_status

    mgr = _get_manager(job_id=job_id)

    if watch:
        watch_status(mgr, job_id, interval=interval)
        return

    info = mgr.detailed_status(job_id)
    print_status(info)


@main.command()
@click.argument('job_id')
@click.option('--lines', '-n', default=50, help='Number of log lines')
def log(job_id, lines):
    """View job log"""
    mgr = _get_manager(job_id=job_id)
    mgr.get_log(job_id, lines=lines)


@main.command()
@click.argument('job_id')
def cancel(job_id):
    """Cancel a running job"""
    mgr = _get_manager(job_id=job_id)
    mgr.cancel(job_id)


@main.command()
@click.argument('job_id')
@click.option('--parse', is_flag=True, help='Parse results and print summary')
def results(job_id, parse):
    """Download job results"""
    mgr = _get_manager(job_id=job_id)

    if hasattr(mgr, 'get_results'):
        result = mgr.get_results(job_id, parse=parse)
        if parse and hasattr(result, 'shape'):
            click.echo(f"\n{result.to_string(max_rows=20)}")
        elif not parse:
            click.echo(f"✓ Results at: {result}")
    elif hasattr(mgr, 'download'):
        result = mgr.download(job_id)
        click.echo(f"✓ Results at: {result}")
    else:
        click.echo("Results download not supported for this job type.")


# ── Submit commands ─────────────────────────────────────────────

@main.command()
@click.argument('query', required=True)
@click.option('--db', 'database', required=True, help='Database alias (e.g., uniref90)')
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--preset', default=None,
              type=click.Choice(['default', 'fast', 'sensitive', 'exhaustive', 'very_sensitive']),
              help='Search preset')
@click.option('--uniprot', is_flag=True,
              help='Treat QUERY as a UniProt accession ID instead of a file path')
def search(query, database, job_name, preset, uniprot):
    """Submit an MMseqs2 search job.

    QUERY is a path to a FASTA file, or a UniProt accession if --uniprot is set.
    """
    if uniprot:
        from ..api.uniprot import fetch_uniprot
        query_file = fetch_uniprot(query)
        if job_name is None:
            job_name = f"{query}_search"
    else:
        query_file = query
        if not Path(query_file).exists():
            raise click.BadParameter(f"File not found: {query_file}", param_hint="'QUERY'")

    if job_name is None:
        job_name = _auto_name_from_pfam(query_file, 'search')

    mgr = _get_manager(job_type='search')
    kwargs = {}
    if preset:
        kwargs['preset'] = preset
    mgr.submit(query_file, database=database, job_name=job_name, **kwargs)


@main.command()
@click.argument('query', required=True)
@click.option('--db', 'database', default='uniprotkb', help='Database alias')
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--no-lineage', is_flag=True, help='Skip lineage parsing')
@click.option('--uniprot', is_flag=True,
              help='Treat QUERY as a UniProt accession ID instead of a file path')
def taxonomy(query, database, job_name, no_lineage, uniprot):
    """Submit an MMseqs2 taxonomy job.

    QUERY is a path to a FASTA file, or a UniProt accession if --uniprot is set.
    """
    if uniprot:
        from ..api.uniprot import fetch_uniprot
        query_file = fetch_uniprot(query)
        if job_name is None:
            job_name = f"{query}_taxonomy"
    else:
        query_file = query
        if not Path(query_file).exists():
            raise click.BadParameter(f"File not found: {query_file}", param_hint="'QUERY'")

    if job_name is None:
        job_name = _auto_name_from_pfam(query_file, 'taxonomy')

    mgr = _get_manager(job_type='taxonomy')
    mgr.submit(query_file, database=database, job_name=job_name,
               tax_lineage=not no_lineage)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--algorithm', '-a', default='clustalo',
              type=click.Choice(['clustalo', 'mafft', 'muscle']),
              help='Alignment algorithm (default: clustalo)')
@click.option('--format', 'output_format', default=None, help='Output format (default depends on algorithm)')
def align(input_file, job_name, algorithm, output_format):
    """Submit a multiple sequence alignment job"""
    if job_name is None:
        job_name = _auto_name_from_pfam(input_file, 'align')

    mgr = _get_manager(job_type='align')
    kwargs = {'algorithm': algorithm}
    if output_format:
        kwargs['output_format'] = output_format
    mgr.submit(input_file, job_name=job_name, **kwargs)


def _get_hmmer_manager():
    """Create an HmmerScan instance using config defaults."""
    from ..config import get_default_connection
    from ..remote.hmmer import HmmerScan
    from ..remote.base import RemoteJobManager
    from fabric import Connection

    defaults = get_default_connection()
    host = defaults.get('host')
    user = defaults.get('user')

    if not host or not user:
        raise click.ClickException(
            "No connection configured. Run 'beak config init' first."
        )

    key_path = defaults.get('key_path')
    if key_path is None:
        key_path = RemoteJobManager._find_ssh_key(None)

    try:
        conn = Connection(
            host=host,
            user=user,
            connect_timeout=10,
            connect_kwargs={"key_filename": str(Path(key_path).expanduser())}
        )
        return conn, HmmerScan(connection=conn)
    except (ConnectionError, TimeoutError, OSError) as e:
        raise click.ClickException(f"Cannot connect to remote server: {e}")


def _auto_name_from_pfam(query_file: str, job_type: str) -> str:
    """Try to generate a job name from the top Pfam hit of the query.

    Returns a name like 'Pkinase_search' on success, or a readable
    fallback like 'search_swift-folding-falcon' if Pfam is unavailable.
    """
    from ..remote.naming import generate_readable_name

    fallback = f"{job_type}_{generate_readable_name()}"

    try:
        _, hmmer = _get_hmmer_manager()
        hits = hmmer.scan(query_file)
        if hits:
            return f"{hits[0]['pfam_name']}_{job_type}"
    except Exception:
        pass

    return fallback


# ── Pfam command ───────────────────────────────────────────────

def _resolve_query_to_fasta(query: str) -> str:
    """Resolve a query argument to a local FASTA file path.

    Accepts:
        - Path to an existing FASTA file
        - UniProt accession (e.g., P0DTC2, A0A0K9RLB5)
        - Raw amino acid sequence string

    Returns:
        Path to a FASTA file (may be a temp file for string/accession inputs)
    """
    import re
    import tempfile

    # 1. Existing file
    if Path(query).exists():
        return query

    # 2. UniProt accession pattern: 6-10 alphanumeric, starts with letter
    #    Matches: P0DTC2, A0A0K9RLB5, Q9Y6K9
    if re.fullmatch(r'[A-Z][A-Z0-9]{4,9}', query, re.IGNORECASE):
        from ..api.uniprot import fetch_uniprot
        try:
            return fetch_uniprot(query)
        except ValueError:
            pass  # Not a valid accession; fall through to treat as sequence

    # 3. Raw sequence string (amino acid characters only)
    cleaned = re.sub(r'\s+', '', query)
    if re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwyx*-]+', cleaned):
        tmp = Path(tempfile.mkdtemp()) / "query.fasta"
        tmp.write_text(f">query\n{cleaned}\n")
        return str(tmp)

    raise click.BadParameter(
        f"'{query}' is not a FASTA file, UniProt accession, or valid protein sequence.",
        param_hint="'QUERY'",
    )


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

    # --lineage implies --taxonomy
    if lineage:
        taxonomy = True

    if pfam_id:
        # Skip scan, use the provided Pfam ID directly
        domains = [{'pfam_id': pfam_id, 'pfam_name': pfam_id,
                     'description': '', 'i_evalue': '-'}]
        if not uniprot:
            uniprot = True  # --pfam without --uniprot makes no sense; assume it
    else:
        # Need a query for scanning
        if not query:
            raise click.UsageError(
                "QUERY is required unless --pfam is provided."
            )

        query_file = _resolve_query_to_fasta(query)
        console.print(f"[brand]Scanning for Pfam domains...[/brand]")
        _, hmmer = _get_hmmer_manager()
        domains = hmmer.scan(query_file, evalue=evalue)

        if not domains:
            console.print("[dim]No Pfam domains found.[/dim]")
            return

        # Display domain table
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

    # Step 2: UniProt lookup (if requested)
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

            # Display results
            header = f"{pid} — {len(df)} UniProt entries"
            result_table = Table(title=header, border_style=BEAK_BLUE)
            result_table.add_column("Accession", style="bold")
            result_table.add_column("Protein Name")

            if taxonomy:
                result_table.add_column("Organism")
                result_table.add_column("Tax ID", justify="right")
            if lineage:
                result_table.add_column("Lineage")

            # Show first 50 rows in CLI, mention total
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

    # Step 3: Structure download (if requested)
    if fetch_structures_flag and uniprot:
        from ..api.structures import find_structures, fetch_structures as fetch_structs
        from rich.status import Status as StructStatus

        # Collect all UniProt IDs from all domains
        all_accessions = []
        for domain in domains:
            pid = domain['pfam_id']
            domain_df = query_uniprot_by_pfam(pid, taxonomy=False)
            all_accessions.extend(domain_df['accession'].tolist())

        if all_accessions:
            # Deduplicate
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

            # Download
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


# ── Structures command ─────────────────────────────────────────

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

    # Collect IDs from arguments and/or file
    ids = list(uniprot_ids)
    if input_file:
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ids.append(line.split()[0])  # take first column

    if not ids:
        raise click.UsageError(
            "Provide UniProt IDs as arguments or via --input file."
        )

    ids = list(dict.fromkeys(ids))  # dedupe

    # Find
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

    # Display summary table
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

    # Fetch
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


# ── Features command ──────────────────────────────────────────

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

    # Collect all .cif files from arguments
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

    # Output
    if output_path:
        if output_format == 'parquet':
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
        console.print(
            f"[green]Wrote {len(result)} rows to {output_path}[/green]"
        )
    else:
        # Display Rich table for terminal output
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


# ── Setup commands ─────────────────────────────────────────────

@main.group()
def setup():
    """Set up databases and tools on the remote server"""
    pass


@setup.command('pfam')
@click.option('--system', is_flag=True,
              help='Install to /srv/protein_sequence_databases/pfam/ (may need sudo)')
@click.option('--path', 'custom_path', default=None,
              help='Custom install path on the remote server')
@click.option('--status', 'show_status', is_flag=True,
              help='Check current Pfam database status')
@click.option('--update', is_flag=True,
              help='Re-download the latest Pfam release')
def setup_pfam(system, custom_path, show_status, update):
    """Download and prepare Pfam-A HMM database on the remote server.

    Default: installs to ~/beak_databases/pfam/ on the remote.
    With --system: installs to /srv/protein_sequence_databases/pfam/ (shared).
    """
    from .theme import get_console, BEAK_BLUE
    from ..remote.hmmer import resolve_pfam_path, PFAM_HMM_FILE
    from ..config import set_config_value
    import datetime

    console = get_console()
    conn, _ = _get_hmmer_manager()

    # --status: just report current state
    if show_status:
        try:
            pfam_path = resolve_pfam_path(conn)
        except FileNotFoundError:
            console.print("[red]Pfam database not found.[/red]")
            console.print("Run [cyan]beak setup pfam[/cyan] to install.")
            return

        # Get age
        age_str = _get_remote_file_age(conn, f"{pfam_path}/{PFAM_HMM_FILE}")

        # Check pressed indices
        pressed = conn.run(
            f'[ -f {pfam_path}/{PFAM_HMM_FILE}.h3i ] && echo YES || echo NO',
            hide=True, warn=True,
        )
        pressed_ok = pressed.stdout.strip() == 'YES'

        # HMMER version
        hmmer_ver = conn.run(
            'hmmscan -h 2>&1 | head -2 | tail -1',
            hide=True, warn=True,
        )

        console.print(f"\n[brand]Pfam Database Status[/brand]")
        console.print(f"  Location:  {pfam_path}/{PFAM_HMM_FILE}")
        console.print(f"  Age:       {age_str}")
        console.print(f"  Pressed:   {'[green]yes[/green]' if pressed_ok else '[red]no[/red]'}")
        if hmmer_ver.ok:
            console.print(f"  HMMER:     {hmmer_ver.stdout.strip()}")
        console.print()
        return

    # Determine target path
    if custom_path:
        target = custom_path
    elif system:
        target = "/srv/protein_sequence_databases/pfam"
    else:
        target = "~/beak_databases/pfam"

    # Expand tilde
    if target.startswith('~'):
        home_result = conn.run('echo $HOME', hide=True)
        target = home_result.stdout.strip() + target[1:]

    # Check if already installed (and not --update)
    existing = conn.run(
        f'[ -f {target}/{PFAM_HMM_FILE} ] && echo EXISTS || echo MISSING',
        hide=True, warn=True,
    )
    if existing.stdout.strip() == 'EXISTS' and not update:
        console.print(f"[green]Pfam database already installed at {target}[/green]")
        console.print("Use [cyan]--update[/cyan] to re-download the latest release.")
        return

    # Check HMMER is available
    hmmer_check = conn.run('command -v hmmscan && command -v hmmpress',
                           hide=True, warn=True)
    if not hmmer_check.ok:
        raise click.ClickException(
            "HMMER not found on the remote server. "
            "Install with: sudo apt install hmmer (or conda install -c bioconda hmmer)"
        )

    # Create directory
    console.print(f"[brand]Setting up Pfam database at {target}[/brand]\n")

    # Use sudo for all file operations when --system
    sudo = 'sudo ' if system else ''

    if system:
        conn.run(f'sudo mkdir -p {target}', hide=True, warn=True)
    else:
        conn.run(f'mkdir -p {target}', hide=True, warn=True)

    # Download
    console.print("  Downloading Pfam-A.hmm.gz ...")
    pfam_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
    dl_result = conn.run(
        f'{sudo}wget -q -O {target}/Pfam-A.hmm.gz {pfam_url}',
        hide=True, warn=True,
    )
    if not dl_result.ok:
        wget_err = (dl_result.stderr or '').strip()
        # Try curl as fallback
        dl_result = conn.run(
            f'{sudo}curl -sfL -o {target}/Pfam-A.hmm.gz {pfam_url}',
            hide=True, warn=True,
        )
    if not dl_result.ok:
        curl_err = (dl_result.stderr or '').strip()
        console.print(f"[dim]  wget: {wget_err or 'failed'}[/dim]")
        console.print(f"[dim]  curl: {curl_err or 'failed'}[/dim]")
        raise click.ClickException(
            f"Download failed from {pfam_url}\n"
            "Check: (1) wget or curl is installed on the remote, "
            "(2) the remote has internet access, "
            "(3) the URL is reachable."
        )

    # Decompress
    console.print("  Decompressing ...")
    gunzip = conn.run(f'{sudo}gunzip -f {target}/Pfam-A.hmm.gz', hide=True, warn=True)
    if not gunzip.ok:
        raise click.ClickException(f"gunzip failed: {gunzip.stderr}")

    # Press
    console.print("  Running hmmpress ...")
    press = conn.run(f'{sudo}hmmpress {target}/Pfam-A.hmm', hide=True, warn=True)
    if not press.ok:
        raise click.ClickException(f"hmmpress failed: {press.stderr}")

    # Fix permissions for --system
    if system:
        conn.run(f'sudo chmod 755 {target}', hide=True, warn=True)
        conn.run(f'sudo chmod 644 {target}/Pfam-A.hmm*', hide=True, warn=True)

    # Save to config
    set_config_value('databases.pfam_path', target)

    console.print(f"\n[green]Pfam database ready at {target}[/green]")
    console.print(f"Path saved to config ([cyan]databases.pfam_path[/cyan])")


def _get_remote_file_age(conn, remote_path: str) -> str:
    """Get human-readable age of a remote file via stat."""
    import datetime

    # Try Linux stat first, then macOS
    result = conn.run(
        f'stat -c %Y {remote_path} 2>/dev/null || stat -f %m {remote_path} 2>/dev/null',
        hide=True, warn=True,
    )
    if not result.ok or not result.stdout.strip():
        return "unknown"

    try:
        mtime = int(result.stdout.strip())
        age_days = int((datetime.datetime.now().timestamp() - mtime) / 86400)
        mod_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

        if age_days == 0:
            return f"today ({mod_date})"
        elif age_days == 1:
            return f"1 day ({mod_date})"
        else:
            age_str = f"{age_days} days ({mod_date})"
            if age_days > 180:
                age_str += " [yellow](consider updating)[/yellow]"
            return age_str
    except (ValueError, OSError):
        return "unknown"


@main.command()
@click.option('--type', 'db_type', default=None,
              type=click.Choice(['search', 'taxonomy']),
              help='Show databases for search or taxonomy')
def databases(db_type):
    """List available remote databases"""
    if db_type == 'taxonomy' or db_type is None:
        mgr = _get_manager(job_type='taxonomy')
        click.echo("Taxonomy databases:")
        df = mgr.list_databases()
        click.echo(df.to_string(index=False))

    if db_type == 'search' or db_type is None:
        if db_type is None:
            click.echo()
        mgr = _get_manager(job_type='search')
        click.echo("Search databases:")
        df = mgr.list_databases()
        click.echo(df.to_string(index=False))


if __name__ == '__main__':
    main()
