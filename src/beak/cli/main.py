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
@click.option('--af2', 'check_af2', is_flag=True,
              help='Also probe the AlphaFold2/ColabFold install. Slower '
                   '(imports JAX to resolve the real backend). Reported '
                   'under its own af2.ok; does not affect the top-level ok.')
@click.pass_context
def doctor(ctx, json_local, check_af2):
    """Check remote server for required tools and databases"""
    from .theme import (get_console, BEAK_BLUE, CATEGORY_STYLES,
                        CATEGORY_LABELS)
    from ._common import get_manager, get_remote_file_age, json_mode, emit_json
    from ..remote.hmmer import resolve_pfam_path, PFAM_HMM_FILE
    from ..remote.af2 import (probe_af2, EXPECTED_CUDNN_LIBS,
                              MIN_WEIGHT_FILES)
    from rich.table import Table

    mgr = get_manager(job_type='search')
    results = mgr.verify_remote(verbose=False)

    af2 = probe_af2(mgr.conn, deep=True) if check_af2 else None

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
        if af2 is not None:
            payload['af2'] = af2
        emit_json(payload)
        # Bare exit code (not ctx.exit / ClickException) so behavior is
        # identical under CliRunner and the real console-script wrapper.
        if not payload['ok']:
            raise SystemExit(1)
        return

    console = get_console()
    console.print(f"\n[brand]BEAK Doctor[/brand]")
    console.print(f"[dim]Remote: {mgr.conn.host}[/dim]\n")

    def section_row(table, category):
        """Category heading row. Rich has no colspan, so the label sits in
        the first column and the rest of the row stays empty."""
        label = CATEGORY_LABELS.get(category, category.title())
        table.add_section()
        table.add_row(f"[cat.{category}]{label}[/cat.{category}]",
                      *[""] * (len(table.columns) - 1))

    def grouped(items, category_of):
        """Yield (category, items) in CATEGORY_STYLES order, skipping empty
        groups and appending any category the palette doesn't know about."""
        buckets = {}
        for key, info in items:
            buckets.setdefault(category_of(info), []).append((key, info))
        ordered = [c for c in CATEGORY_STYLES if c in buckets]
        ordered += [c for c in buckets if c not in CATEGORY_STYLES]
        return [(c, buckets[c]) for c in ordered]

    tools_table = Table(title="Tools", border_style=BEAK_BLUE, show_lines=False)
    tools_table.add_column("Tool", style="bold")
    tools_table.add_column("Status")
    tools_table.add_column("Version", style="dim")
    tools_table.add_column("Needed By", style="dim")

    for category, tools in grouped(results['tools'].items(),
                                   lambda i: i.get('category', 'other')):
        section_row(tools_table, category)
        for tool, info in tools:
            if info['found']:
                status = "[green]OK[/green]"
                version = info.get('version', '') or ''
            elif info['required']:
                status = "[red]MISSING[/red]"
                version = info.get('install', '')
            else:
                status = "[dim]--[/dim]"
                version = ''
            tools_table.add_row(f"  {tool}", status, version[:50],
                                info['needed_by'])

    console.print(tools_table)
    console.print()

    db_info = results.get('databases', {})
    entries = db_info.get('entries', {})

    db_table = Table(title="Databases", border_style=BEAK_BLUE, show_lines=False)
    db_table.add_column("Database", style="bold")
    db_table.add_column("Status")
    db_table.add_column("Notes", style="dim")
    db_table.add_column("File", style="dim")

    # Sequence databases, grouped by molecule type. An absent database is
    # rendered dim rather than red — most sites install a subset, so missing
    # is normal here and does not flip the overall `ok`.
    for category, group in grouped(
            sorted(entries.items()),
            lambda i: f"sequence.{i.get('molecule', 'protein')}"):
        section_row(db_table, category)
        for alias, info in group:
            db_table.add_row(
                f"  {alias}",
                "[green]OK[/green]" if info['found'] else "[dim]--[/dim]",
                "taxonomy" if info.get('has_taxonomy') else "",
                info['name'],
            )

    if not entries and not db_info.get('exists'):
        section_row(db_table, "sequence.protein")
        db_table.add_row("  MMseqs2", "[red]MISSING[/red]", "",
                         "[dim]directory not found[/dim]")

    section_row(db_table, "profile")
    try:
        pfam_path = resolve_pfam_path(mgr.conn)
        age_str = get_remote_file_age(mgr.conn, f"{pfam_path}/{PFAM_HMM_FILE}")
        db_table.add_row("  Pfam-A", "[green]OK[/green]", age_str, pfam_path)
    except FileNotFoundError:
        db_table.add_row(
            "  Pfam-A",
            "[dim]--[/dim]",
            "",
            "[dim]not installed (beak setup pfam)[/dim]",
        )

    console.print(db_table)
    if entries:
        n_found = sum(1 for e in entries.values() if e['found'])
        # highlight=False: the default highlighter would repaint the counts
        # and the path, which fights the dim-footnote role of this line.
        console.print(
            f"[dim]{n_found} of {len(entries)} known sequence databases "
            f"under {db_info.get('path', '?')} · "
            f"`beak databases` for sizes[/dim]",
            highlight=False,
        )

    if af2 is not None:
        def _mark(good: bool, warn_only: bool = False) -> str:
            if good:
                return "[green]OK[/green]"
            return "[yellow]WARN[/yellow]" if warn_only else "[red]FAIL[/red]"

        af2_table = Table(title="AlphaFold2 / ColabFold",
                          border_style=BEAK_BLUE, show_lines=False)
        af2_table.add_column("Check", style="bold")
        af2_table.add_column("Status")
        af2_table.add_column("Detail", style="dim")

        if not af2['installed']:
            af2_table.add_row("colabfold_batch", "[dim]--[/dim]",
                              "not found (see beak doctor --af2 help)")
        else:
            af2_table.add_row("colabfold_batch", _mark(True),
                              f"{af2['binary']} [{af2['binary_source']}]")
            af2_table.add_row("Weights",
                              _mark(af2['weights'] >= MIN_WEIGHT_FILES),
                              f"{af2['weights']} params files")

            if not af2['numpy_isolated']:
                np_detail = "could not import numpy"
            elif (af2['numpy_default']
                    and af2['numpy_default'] != af2['numpy_isolated']):
                np_detail = (f"env numpy {af2['numpy_isolated']} shadowed by "
                             f"user-site {af2['numpy_default']}")
            else:
                np_detail = f"numpy {af2['numpy_isolated']}"
            af2_table.add_row("Starts up",
                              _mark(af2['binary_runs'] == 'ok'), np_detail)

            if af2['gpu_total']:
                af2_table.add_row(
                    "cuDNN override",
                    _mark(af2['cudnn_libs'] == EXPECTED_CUDNN_LIBS,
                          warn_only=True),
                    f"{af2['cudnn_libs']}/{EXPECTED_CUDNN_LIBS} libraries")
                # VRAM is the binding constraint on what will fit, so show
                # it: ColabFold oversubscribes ~4x into host RAM, which
                # stretches a small card but does not remove the limit.
                gpu_detail = f"{af2['gpu_free']} free of {af2['gpu_total']}"
                if af2['gpu_mem_mb']:
                    gb = af2['gpu_mem_mb'] / 1024
                    gpu_detail += (f" · {gb:.0f} GB VRAM "
                                   f"(~{gb * 4:.0f} GB with unified memory)")
                af2_table.add_row("GPUs", _mark(af2['gpu_free'] > 0,
                                                warn_only=True), gpu_detail)
                # Both readings matter: `bare` is what an unconfigured caller
                # gets, `override` is the ceiling. Equal and gpu = healthy.
                bare = af2['backend_bare'] or 'unknown'
                override = af2['backend_override']
                backend_detail = (f"{bare} bare, {override} with override"
                                  if override and override != bare else bare)
                af2_table.add_row(
                    "JAX backend",
                    _mark((af2['backend_override'] or af2['backend_bare'])
                          == 'gpu'),
                    backend_detail)

        console.print()
        console.print(af2_table)

        # Issues carry their own remedy — the whole point of the probe is
        # that these failures are otherwise silent.
        for problem in af2['issues']:
            color = 'red' if problem['level'] == 'error' else 'yellow'
            console.print(f"  [{color}]{problem['level']}[/{color}]  "
                          f"{problem['message']}")
            console.print(f"        [dim]fix: {problem['fix']}[/dim]")

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
