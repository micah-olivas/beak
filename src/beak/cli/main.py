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
    db_table.add_column("Path", style="dim")

    db_info = results.get('databases', {})
    if db_info.get('exists'):
        db_count = db_info.get('count', 0)
        db_table.add_row(
            f"MMseqs2 ({db_count} dbs)",
            "[green]OK[/green]",
            "[dim]--[/dim]",
            db_info['path'],
        )
    else:
        db_table.add_row(
            "MMseqs2",
            "[red]MISSING[/red]",
            "[dim]--[/dim]",
            "[dim]directory not found[/dim]",
        )

    try:
        pfam_path = resolve_pfam_path(mgr.conn)
        age_str = get_remote_file_age(mgr.conn, f"{pfam_path}/{PFAM_HMM_FILE}")
        db_table.add_row("Pfam-A", "[green]OK[/green]", age_str, pfam_path)
    except FileNotFoundError:
        db_table.add_row(
            "Pfam-A",
            "[dim]--[/dim]",
            "[dim]--[/dim]",
            "[dim]not installed (beak setup pfam)[/dim]",
        )

    console.print(db_table)

    disk = results.get('disk', {})
    if disk:
        avail = disk.get('available', '?')
        total = disk.get('total', '?')
        pct_str = disk.get('used_pct') or ''
        try:
            used_pct = int(pct_str.rstrip('%'))
            free_pct = 100 - used_pct
        except (TypeError, ValueError):
            used_pct = None

        if used_pct is None:
            console.print(f"\n[dim]Disk: {avail} available of {total} total[/dim]")
        else:
            # Tiny inline bar — used segment carries one accent color that
            # shifts from brand-blue → amber → red as free space tightens.
            # Free segment stays dim so the bar reads as background.
            bar_width = 14
            filled = max(1, min(bar_width - 1, round(bar_width * used_pct / 100)))
            empty = bar_width - filled
            if free_pct < 5:
                accent = "red"
            elif free_pct < 15:
                accent = "yellow"
            else:
                accent = BEAK_BLUE
            bar = f"[{accent}]{'█' * filled}[/]{'░' * empty}"
            console.print(
                f"\n[dim]Disk[/dim]  {bar}  "
                f"[dim]{avail} free of {total} · {free_pct}% free[/dim]"
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
