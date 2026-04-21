"""Root CLI group and the `doctor` command."""

import click


@click.group()
@click.version_option(package_name='beak')
def main():
    """BEAK - Biophysical and Evolutionary Analysis Kit"""
    pass


@main.command()
def doctor():
    """Check remote server for required tools and databases"""
    from .theme import get_console, BEAK_BLUE
    from ._common import get_manager, get_remote_file_age
    from ..remote.hmmer import resolve_pfam_path, PFAM_HMM_FILE
    from rich.table import Table

    console = get_console()
    mgr = get_manager(job_type='search')
    results = mgr.verify_remote(verbose=False)

    console.print(f"\n[brand]BEAK Doctor[/brand]")
    console.print(f"[dim]Remote: {mgr.conn.host}[/dim]\n")

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

    db_table = Table(title="Databases", border_style=BEAK_BLUE, show_lines=False)
    db_table.add_column("Database", style="bold")
    db_table.add_column("Status")
    db_table.add_column("Age")

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

    try:
        pfam_path = resolve_pfam_path(mgr.conn)
        age_str = get_remote_file_age(mgr.conn, f"{pfam_path}/{PFAM_HMM_FILE}")
        db_table.add_row("Pfam-A", "[green]OK[/green]", age_str)
    except FileNotFoundError:
        db_table.add_row(
            "Pfam-A",
            "[dim]--[/dim]",
            "[dim]not installed (beak setup pfam)[/dim]",
        )

    console.print(db_table)

    disk = results.get('disk', {})
    if disk:
        console.print(
            f"\n[dim]Disk: {disk.get('available', '?')} available "
            f"of {disk.get('total', '?')} total[/dim]"
        )

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


if __name__ == '__main__':
    main()
