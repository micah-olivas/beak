"""Root CLI group and the `doctor` command."""

import click


@click.group()
@click.option('--json', 'json_mode', is_flag=True,
              help='Emit machine-readable JSON on stdout instead of Rich output.')
@click.version_option(package_name='beak')
@click.pass_context
def main(ctx, json_mode):
    """BEAK - Biophysical and Evolutionary Analysis Kit"""
    ctx.ensure_object(dict)
    ctx.obj['json'] = json_mode


@main.command()
@click.option('--json', 'json_local', is_flag=True,
              help='Emit the environment report as a JSON object on stdout.')
@click.pass_context
def doctor(ctx, json_local):
    """Check remote server for required tools and databases"""
    from .theme import get_console, BEAK_BLUE
    from ._common import get_manager, get_remote_file_age, json_mode, emit_json
    from ..remote.hmmer import resolve_pfam_path, PFAM_HMM_FILE
    from rich.table import Table

    mgr = get_manager(job_type='search')
    results = mgr.verify_remote(verbose=False)

    if json_mode(ctx, json_local):
        # Preflight payload: an agent gates submission on `ok` (and the
        # nonzero exit) before spending remote compute.
        payload = {
            'ok': bool(results.get('ok')),
            'remote_host': mgr.conn.host,
            'tools': results.get('tools', {}),
            'databases': results.get('databases', {}),
            'disk': results.get('disk', {}),
            'load': results.get('load', {}),
        }
        try:
            payload['pfam'] = {'installed': True,
                               'path': resolve_pfam_path(mgr.conn)}
        except FileNotFoundError:
            payload['pfam'] = {'installed': False, 'path': None}
        emit_json(payload)
        # Bare exit code (not ctx.exit / ClickException) so behavior is
        # identical under CliRunner and the real console-script wrapper.
        if not payload['ok']:
            raise SystemExit(1)
        return

    console = get_console()
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

    load = results.get('load', {})
    if load.get('load_per_cpu') is not None:
        lpc = load['load_per_cpu']
        # Green under half-loaded, amber approaching saturation, red over.
        accent = "red" if lpc >= 1.0 else "yellow" if lpc >= 0.7 else "green"
        line = (f"[dim]Load[/dim]  [{accent}]{lpc:g}/cpu[/]  "
                f"[dim]({load['load_1m']:g} over {load['n_cpus']} cpus)[/dim]")
        if load.get('mem_available_mb') is not None:
            line += (f"  [dim]· {load['mem_available_mb'] // 1024} GB "
                     f"of {load['mem_total_mb'] // 1024} GB free[/dim]")
        if load.get('gpus'):
            busy = sum(1 for g in load['gpus'] if g['util_pct'] >= 50)
            line += f"  [dim]· {len(load['gpus'])} GPU, {busy} busy[/dim]"
        console.print(line)

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
