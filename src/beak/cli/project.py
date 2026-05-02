"""`beak project` commands."""

from datetime import datetime

import click

from .main import main


@main.group()
def project():
    """Manage beak projects (~/.beak/projects/<name>/)"""
    pass


@project.command('init')
@click.argument('name')
@click.option('--uniprot', 'uniprot_id', default=None,
              help='UniProt accession (e.g., P00533) — fetches sequence + metadata.')
@click.option('--sequence', 'sequence_file', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Local FASTA file with the target sequence.')
@click.option('--description', default="", help='One-line project description.')
def project_init(name, uniprot_id, sequence_file, description):
    """Create a new project with a target sequence."""
    from ..project import BeakProject, BeakProjectError
    from .theme import get_console

    console = get_console()

    try:
        proj = BeakProject.init(
            name=name,
            uniprot_id=uniprot_id,
            sequence_file=sequence_file,
            description=description,
        )
    except BeakProjectError as e:
        raise click.ClickException(str(e))

    target = proj.manifest().get('target', {})
    console.print(f"\n[brand]Created project '{proj.name}'[/brand]")
    console.print(f"[dim]{proj.path}[/dim]\n")
    for label, key in (('Target', 'uniprot_id'), ('Gene', 'gene_name'),
                       ('Organism', 'organism')):
        if target.get(key):
            console.print(f"  {label:9s} {target[key]}")
    console.print(f"  {'Length':9s} {target.get('length', '?')} aa\n")


@project.command('list')
def project_list():
    """List all known projects."""
    from rich.table import Table
    from ..project import BeakProject, PROJECTS_DIR
    from .theme import get_console, BEAK_BLUE

    console = get_console()
    projects = BeakProject.list_projects()

    if not projects:
        console.print(f"\n[dim]No projects yet under {PROJECTS_DIR}.[/dim]")
        console.print("[dim]Create one with: beak project init <name> --uniprot <id>[/dim]\n")
        return

    status_colors = {
        'ready': 'green', 'running': 'yellow', 'submitted': 'cyan',
        'queued': 'cyan', 'completed': 'green', 'failed': 'red',
        'cancelled': 'dim', 'new': 'dim',
    }

    table = Table(title="Projects", border_style=BEAK_BLUE)
    table.add_column("Name", style="bold")
    table.add_column("Target", style="dim")
    table.add_column("Length", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for proj in projects:
        m = proj.manifest()
        target = m.get('target', {})
        proj_meta = m.get('project', {})
        target_id = target.get('uniprot_id') or target.get('uniprot_name') or '-'
        length = target.get('length')
        length_str = f"{length:,}" if length else '-'
        size_str = _human_size(proj.cached_size())
        status = proj.status_summary()
        status_str = f"[{status_colors.get(status, 'dim')}]{status}[/{status_colors.get(status, 'dim')}]"
        created_str = _format_dt(proj_meta.get('created_at'))
        table.add_row(proj.name, target_id, length_str, size_str, status_str, created_str)

    console.print()
    console.print(table)
    console.print()


@project.command('status')
@click.argument('name')
def project_status(name):
    """Show layer state and disk usage for a project."""
    from rich.table import Table
    from ..project import BeakProject, BeakProjectError
    from .theme import get_console, BEAK_BLUE, STAGE_ICONS

    console = get_console()
    try:
        proj = BeakProject.load(name)
    except BeakProjectError as e:
        raise click.ClickException(str(e))

    m = proj.manifest()
    target = m.get('target', {})
    proj_meta = m.get('project', {})

    console.print(f"\n[brand]Project: {proj.name}[/brand]")
    console.print(f"[dim]{proj.path}[/dim]")
    if proj_meta.get('description'):
        console.print(f"[dim]{proj_meta['description']}[/dim]")
    console.print()

    if target:
        console.print("[bold]Target[/bold]")
        for key in ('uniprot_id', 'gene_name', 'organism', 'length'):
            val = target.get(key)
            if val is not None:
                console.print(f"  {key:12s} {val}")
        console.print()

    sizes = proj.disk_usage_by_layer()

    layers_table = Table(title="Layers", border_style=BEAK_BLUE)
    layers_table.add_column("Layer", style="bold")
    layers_table.add_column("Status")
    layers_table.add_column("Size", justify="right")

    for layer in ('target', 'homologs', 'domains', 'structures', 'experiments'):
        present = bool(m.get(layer))
        icon = STAGE_ICONS['done'] if present else STAGE_ICONS['pending']
        size = sizes.get(layer, 0)
        size_str = _human_size(size) if size else '[dim]--[/dim]'
        layers_table.add_row(layer, icon, size_str)

    console.print(layers_table)
    console.print(f"\n[dim]Total on disk: {_human_size(proj.disk_usage())}[/dim]\n")


def _human_size(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ('KB', 'MB', 'GB', 'TB'):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def _format_dt(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    try:
        return datetime.fromisoformat(str(value)).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return str(value)
