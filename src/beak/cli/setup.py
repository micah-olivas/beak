"""Setup commands for remote databases, plus `databases` listing."""

import click

from .main import main
from ._common import get_manager, get_hmmer_manager, get_remote_file_age


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
    from .theme import get_console
    from ..remote.hmmer import resolve_pfam_path, PFAM_HMM_FILE
    from ..config import set_config_value

    console = get_console()
    conn, _ = get_hmmer_manager()

    if show_status:
        try:
            pfam_path = resolve_pfam_path(conn)
        except FileNotFoundError:
            console.print("[red]Pfam database not found.[/red]")
            console.print("Run [cyan]beak setup pfam[/cyan] to install.")
            return

        age_str = get_remote_file_age(conn, f"{pfam_path}/{PFAM_HMM_FILE}")

        pressed = conn.run(
            f'[ -f {pfam_path}/{PFAM_HMM_FILE}.h3i ] && echo YES || echo NO',
            hide=True, warn=True,
        )
        pressed_ok = pressed.stdout.strip() == 'YES'

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

    if custom_path:
        target = custom_path
    elif system:
        target = "/srv/protein_sequence_databases/pfam"
    else:
        target = "~/beak_databases/pfam"

    if target.startswith('~'):
        home_result = conn.run('echo $HOME', hide=True)
        target = home_result.stdout.strip() + target[1:]

    existing = conn.run(
        f'[ -f {target}/{PFAM_HMM_FILE} ] && echo EXISTS || echo MISSING',
        hide=True, warn=True,
    )
    if existing.stdout.strip() == 'EXISTS' and not update:
        console.print(f"[green]Pfam database already installed at {target}[/green]")
        console.print("Use [cyan]--update[/cyan] to re-download the latest release.")
        return

    hmmer_check = conn.run('command -v hmmscan && command -v hmmpress',
                           hide=True, warn=True)
    if not hmmer_check.ok:
        raise click.ClickException(
            "HMMER not found on the remote server. "
            "Install with: sudo apt install hmmer (or conda install -c bioconda hmmer)"
        )

    console.print(f"[brand]Setting up Pfam database at {target}[/brand]\n")

    sudo = 'sudo ' if system else ''

    if system:
        conn.run(f'sudo mkdir -p {target}', hide=True, warn=True)
    else:
        conn.run(f'mkdir -p {target}', hide=True, warn=True)

    console.print("  Downloading Pfam-A.hmm.gz ...")
    pfam_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
    dl_result = conn.run(
        f'{sudo}wget -q -O {target}/Pfam-A.hmm.gz {pfam_url}',
        hide=True, warn=True,
    )
    if not dl_result.ok:
        wget_err = (dl_result.stderr or '').strip()
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

    console.print("  Decompressing ...")
    gunzip = conn.run(f'{sudo}gunzip -f {target}/Pfam-A.hmm.gz', hide=True, warn=True)
    if not gunzip.ok:
        raise click.ClickException(f"gunzip failed: {gunzip.stderr}")

    console.print("  Running hmmpress ...")
    press = conn.run(f'{sudo}hmmpress {target}/Pfam-A.hmm', hide=True, warn=True)
    if not press.ok:
        raise click.ClickException(f"hmmpress failed: {press.stderr}")

    if system:
        conn.run(f'sudo chmod 755 {target}', hide=True, warn=True)
        conn.run(f'sudo chmod 644 {target}/Pfam-A.hmm*', hide=True, warn=True)

    set_config_value('databases.pfam_path', target)

    console.print(f"\n[green]Pfam database ready at {target}[/green]")
    console.print(f"Path saved to config ([cyan]databases.pfam_path[/cyan])")


@main.command()
@click.option('--type', 'db_type', default=None,
              type=click.Choice(['search', 'taxonomy']),
              help='Show databases for search or taxonomy')
def databases(db_type):
    """List available remote databases"""
    if db_type == 'taxonomy' or db_type is None:
        mgr = get_manager(job_type='taxonomy')
        click.echo("Taxonomy databases:")
        df = mgr.list_databases()
        click.echo(df.to_string(index=False))

    if db_type == 'search' or db_type is None:
        if db_type is None:
            click.echo()
        mgr = get_manager(job_type='search')
        click.echo("Search databases:")
        df = mgr.list_databases()
        click.echo(df.to_string(index=False))
