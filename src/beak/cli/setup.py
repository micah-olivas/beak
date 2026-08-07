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


@setup.command('af2')
@click.option('--home', 'af2_home', default=None,
              help='Path to the localcolabfold install on the remote. '
                   'Recorded in config so other commands reuse it.')
@click.option('--cudnn-dir', 'cudnn_dir', default=None,
              help='Directory holding the cuDNN 8.9 override libraries. '
                   'Point several users at one shared copy to avoid '
                   'duplicating ~700 MB each.')
@click.option('--status', 'show_status', is_flag=True,
              help='Report what is installed without changing anything')
@click.option('--dry-run', is_flag=True,
              help='Print the plan without executing it')
def setup_af2(af2_home, cudnn_dir, show_status, dry_run):
    """Configure a remote AlphaFold2 / localcolabfold install for beak.

    Discovers an existing install and cuDNN override, reusing a shared copy
    when one exists, then writes per-user wrapper scripts that carry the
    environment fixes AF2 needs (PYTHONNOUSERSITE, the cuDNN LD_PRELOAD, and
    no GPU preallocation).

    Everything it does is additive and confined to your home directory. It
    never uses sudo and never deletes anything.
    """
    from .theme import get_console
    from ..remote.af2 import (discover_af2, plan_af2_setup, apply_af2_setup,
                              probe_af2, WRAPPER_ENTRY_POINTS)
    from ..config import set_config_value

    console = get_console()
    mgr = get_manager(job_type='search')

    if not af2_home or not cudnn_dir:
        from ..config import load_config
        section = load_config().get('af2', {})
        af2_home = af2_home or section.get('home')
        cudnn_dir = cudnn_dir or section.get('cudnn_dir')

    discovery = discover_af2(mgr.conn, af2_home=af2_home, cudnn_dir=cudnn_dir)

    console.print("\n[brand]AlphaFold2 / ColabFold setup[/brand]")
    console.print(f"[dim]Remote: {mgr.conn.host}[/dim]\n")
    console.print(f"  install:  {discovery['af2_home'] or 'not found'}")
    console.print(f"  cuDNN:    {discovery['cudnn_dir'] or 'not found'}")
    version = discovery['wrapper_version']
    console.print(f"  wrappers: "
                  f"{'v%d' % version if version else 'not installed'}")

    if show_status:
        report = probe_af2(mgr.conn, af2_home=discovery['af2_home'],
                           cudnn_dir=discovery['cudnn_dir'], deep=False)
        console.print(f"\n  starts up: "
                      f"{'yes' if report['binary_runs'] == 'ok' else 'no'}")
        for problem in report['issues']:
            console.print(f"  [yellow]{problem['level']}[/yellow] "
                          f"{problem['message']}")
        return

    plan = plan_af2_setup(discovery)
    if plan['blocked']:
        console.print(f"\n[red]Cannot proceed:[/red] {plan['blocked']}")
        raise SystemExit(1)

    if not plan['steps']:
        console.print("\n[green]Already set up — nothing to do.[/green]\n")
        return

    console.print("\n[bold]Plan[/bold]")
    for step in plan['steps']:
        console.print(f"  · {step['detail']}")

    if dry_run:
        console.print("\n[dim]--dry-run: nothing executed.[/dim]\n")
        return

    console.print()
    results = apply_af2_setup(
        mgr.conn, plan,
        on_step=lambda msg: console.print(f"  [dim]{msg}…[/dim]"))

    console.print()
    failed = [r for r in results if not r['ok']]
    for result in results:
        mark = "[green]OK[/green]" if result['ok'] else "[red]FAILED[/red]"
        console.print(f"  {mark}  {result['action']}: {result['message']}")

    if failed:
        console.print("\n[red]Setup incomplete.[/red]\n")
        raise SystemExit(1)

    # Persist what we resolved so `doctor --af2` and any future AF2 command
    # reuse the same paths instead of re-discovering them.
    set_config_value('af2.home', plan['af2_home'])
    set_config_value('af2.cudnn_dir', plan['cudnn_dir'])

    console.print(f"\n[green]Done.[/green] Wrappers: "
                  f"{', '.join(WRAPPER_ENTRY_POINTS)} in ~/bin")
    console.print("[dim]Verify with: beak doctor --af2[/dim]\n")


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
