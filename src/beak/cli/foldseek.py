"""Local structural homology search via foldseek.

Unlike the sequence-search commands (which submit remote SSH jobs),
``foldseek`` runs locally against a downloaded structure database:

    beak foldseek setup --db Alphafold/Swiss-Prot   # one-time db download
    beak foldseek search --project my_kinase        # search the target
    beak foldseek search --query model.cif -o hits.parquet
"""

import click
from pathlib import Path

from .main import main


# Root under which `beak foldseek setup --db NAME` installs databases.
_DB_ROOT = Path.home() / ".beak" / "foldseek_db"


def _sanitize_db_name(db_name: str) -> str:
    """Turn a foldseek db name into a safe directory name."""
    return db_name.replace("/", "_").replace(" ", "_")


@main.group()
def foldseek():
    """Local structural homology search (foldseek)."""
    pass


@foldseek.command('setup')
@click.option('--db', 'db_name', default=None,
              help='Prebuilt database to download (e.g. PDB, '
                   'Alphafold/Swiss-Prot, Alphafold/Proteome).')
@click.option('--path', 'install_path', default=None,
              help='Directory to install the downloaded database into '
                   '(default: ~/.beak/foldseek_db/<db>).')
@click.option('--db-path', 'existing_db', default=None,
              help='Point at an already-built database prefix instead of '
                   'downloading.')
@click.option('--binary', 'binary_path', default=None,
              help='Path to the foldseek executable (recorded in config).')
@click.option('--status', 'show_status', is_flag=True,
              help='Show the current foldseek configuration and exit.')
def foldseek_setup(db_name, install_path, existing_db, binary_path,
                   show_status):
    """Configure foldseek: download a database, or point at an existing one.

    Examples:

        beak foldseek setup --status

        beak foldseek setup --db PDB

        beak foldseek setup --db-path /data/foldseek/afdb/db

        beak foldseek setup --binary /opt/foldseek/bin/foldseek
    """
    from .theme import get_console
    from ..config import get_foldseek_config, set_config_value
    from ..structures.foldseek import (
        KNOWN_DATABASES, FoldseekError, download_database,
        foldseek_version, resolve_foldseek_binary,
    )

    console = get_console()

    # --binary: record the executable path first so later steps use it.
    if binary_path:
        try:
            resolved = resolve_foldseek_binary(binary_path)
        except FoldseekError as e:
            raise click.ClickException(str(e))
        set_config_value('foldseek.binary', resolved)
        console.print(f"[green]foldseek binary set to {resolved}[/green]")

    if show_status or (not db_name and not existing_db and not binary_path):
        cfg = get_foldseek_config()
        try:
            resolved = resolve_foldseek_binary()
        except FoldseekError:
            resolved = None
        version = foldseek_version() if resolved else None

        console.print("\n[brand]foldseek configuration[/brand]")
        console.print(
            f"  Binary:    {resolved or '[red]not found[/red]'}"
            + (f"  [dim]({version})[/dim]" if version else "")
        )
        console.print(f"  Database:  {cfg.get('db_path') or '[dim]not set[/dim]'}")
        if cfg.get('db_name'):
            console.print(f"  DB name:   {cfg['db_name']}")
        if not cfg.get('db_path'):
            console.print(
                "\nDownload one with "
                "[cyan]beak foldseek setup --db PDB[/cyan], or point at an "
                "existing prefix with [cyan]--db-path[/cyan]."
            )
        console.print(
            f"\n[dim]Known databases: {', '.join(KNOWN_DATABASES)}[/dim]"
        )
        console.print()
        return

    # --db-path: adopt an existing database prefix, no download.
    if existing_db:
        prefix = Path(existing_db).expanduser()
        parent = prefix.parent
        if not parent.is_dir() or not any(parent.glob(prefix.name + "*")):
            raise click.ClickException(
                f"No foldseek database found at prefix '{existing_db}'. "
                "Point --db-path at the database prefix "
                "(e.g. /data/foldseek/afdb/db)."
            )
        set_config_value('foldseek.db_path', str(prefix))
        console.print(f"[green]foldseek database set to {prefix}[/green]")
        console.print("Path saved to config ([cyan]foldseek.db_path[/cyan])")
        return

    # --db: download a prebuilt database.
    try:
        resolve_foldseek_binary()
    except FoldseekError as e:
        raise click.ClickException(str(e))

    if db_name not in KNOWN_DATABASES:
        console.print(
            f"[dim]'{db_name}' is not in the known list "
            f"({', '.join(KNOWN_DATABASES)}); passing it to foldseek "
            "anyway.[/dim]"
        )

    dest = (Path(install_path).expanduser() if install_path
            else _DB_ROOT / _sanitize_db_name(db_name))
    console.print(
        f"[brand]Downloading {db_name} into {dest}[/brand]\n"
        "[dim]This can be large and slow; foldseek's progress follows.[/dim]\n"
    )
    try:
        prefix = download_database(db_name, str(dest))
    except FoldseekError as e:
        raise click.ClickException(str(e))

    set_config_value('foldseek.db_path', str(prefix))
    set_config_value('foldseek.db_name', db_name)
    console.print(f"\n[green]Database ready at {prefix}[/green]")
    console.print("Path saved to config ([cyan]foldseek.db_path[/cyan])")


@foldseek.command('search')
@click.option('--project', 'project_name', default=None,
              help='Search the target structure of this project and record '
                   'the results as a project layer.')
@click.option('--query', 'query_path', type=click.Path(exists=True),
              default=None,
              help='Query structure (mmCIF/PDB). Overrides --project target.')
@click.option('--db', 'db_path', default=None,
              help='Target database prefix (default: foldseek.db_path config).')
@click.option('--sensitivity', '-s', type=float, default=None,
              help='Foldseek sensitivity (default: foldseek default, 9.5).')
@click.option('--evalue', '-e', type=float, default=None,
              help='E-value threshold (default: foldseek default, 0.001).')
@click.option('--max-seqs', type=int, default=None,
              help='Max target hits to report (foldseek --max-seqs).')
@click.option('--threads', type=int, default=None,
              help='CPU threads for foldseek.')
@click.option('--output', '-o', 'output_path', default=None,
              help='Write hits to this CSV/Parquet (non-project mode).')
@click.option('--json', 'json_local', is_flag=True,
              help='Emit results as a JSON object on stdout.')
@click.pass_context
def foldseek_search(ctx, project_name, query_path, db_path, sensitivity,
                    evalue, max_seqs, threads, output_path, json_local):
    """Run a local foldseek structural search.

    Point it at a project (uses the target's cached structure and records
    a `structural` layer) or at an arbitrary --query structure.

    Examples:

        beak foldseek search --project my_kinase

        beak foldseek search --query model.cif -s 9.5 -o hits.parquet
    """
    from datetime import datetime
    from .theme import get_console, BEAK_BLUE
    from ._common import json_mode, emit_json
    from ..config import get_foldseek_config
    from ..structures.foldseek import (
        FoldseekError, foldseek_version, run_easy_search,
    )

    console = get_console()
    as_json = json_mode(ctx, json_local)

    # Resolve the database.
    db = db_path or get_foldseek_config().get('db_path')
    if not db:
        raise click.ClickException(
            "No foldseek database configured. Run "
            "'beak foldseek setup --db PDB' or pass --db PREFIX."
        )

    # Resolve the query structure + (optional) project handle.
    proj = None
    query_source = None
    if query_path:
        query = Path(query_path)
        query_source = str(query)
    elif project_name:
        from ..project import BeakProject, BeakProjectError
        try:
            proj = BeakProject.load(project_name)
        except BeakProjectError as e:
            raise click.ClickException(str(e))
        query = _resolve_project_query(proj)
        query_source = "target"
    else:
        raise click.UsageError("Provide --project NAME or --query FILE.")

    if not as_json:
        console.print(
            f"[brand]foldseek search[/brand]  "
            f"[dim]{query.name} vs {db}[/dim]"
        )

    # In project mode write the raw m8 durably alongside the hits table.
    out_m8 = None
    if proj is not None:
        proj.structural_dir.mkdir(parents=True, exist_ok=True)
        out_m8 = str(proj.structural_dir / "raw.m8")

    from rich.status import Status
    status = None
    if not as_json:
        status = Status("Running foldseek...", console=console)
        status.start()
    try:
        hits = run_easy_search(
            str(query), db,
            out_path=out_m8,
            sensitivity=sensitivity,
            evalue=evalue,
            max_seqs=max_seqs,
            threads=threads,
        )
    except FoldseekError as e:
        raise click.ClickException(str(e))
    finally:
        if status is not None:
            status.stop()

    n_hits = len(hits)

    # Persist: project layer, or standalone -o file.
    hits_path = None
    if proj is not None:
        hits_path = str(proj.structural_dir / "hits.parquet")
        hits.to_parquet(hits_path, index=False)
        layer = {
            "db_path": str(db),
            "db_name": get_foldseek_config().get('db_name'),
            "query": query.name,
            "query_source": query_source,
            "n_hits": int(n_hits),
            "sensitivity": sensitivity,
            "evalue": evalue,
            "max_seqs": max_seqs,
            "foldseek_version": foldseek_version(),
            "last_updated": datetime.now(),
        }
        # TOML can't serialize None — drop unset optional fields.
        layer = {k: v for k, v in layer.items() if v is not None}
        with proj.mutate() as m:
            m["structural"] = layer
    elif output_path:
        if output_path.endswith(".parquet"):
            hits.to_parquet(output_path, index=False)
        else:
            hits.to_csv(output_path, index=False)
        hits_path = output_path

    if as_json:
        emit_json({
            "query": str(query),
            "query_source": query_source,
            "db": str(db),
            "n_hits": int(n_hits),
            "hits_path": hits_path,
            "raw_m8": out_m8,
            "hits": hits.to_dict(orient="records"),
        })
        return

    if n_hits == 0:
        console.print("[dim]No structural hits found.[/dim]")
        return

    from rich.table import Table
    import pandas as pd

    table = Table(
        title=f"Structural hits ({n_hits})", border_style=BEAK_BLUE,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Target", style="bold")
    table.add_column("TM-score", justify="right")
    table.add_column("LDDT", justify="right")
    table.add_column("Ident", justify="right")
    table.add_column("E-value", justify="right")
    table.add_column("Qcov", justify="right")

    def _fmt(val, spec):
        return format(val, spec) if pd.notna(val) else "-"

    display_limit = 50
    for i, (_, row) in enumerate(hits.head(display_limit).iterrows(), 1):
        table.add_row(
            str(i),
            str(row.get("target", "-")),
            _fmt(row.get("alntmscore"), ".3f"),
            _fmt(row.get("lddt"), ".3f"),
            _fmt(row.get("fident"), ".2f"),
            _fmt(row.get("evalue"), ".1e"),
            _fmt(row.get("qcov"), ".2f"),
        )

    console.print(table)
    if n_hits > display_limit:
        console.print(f"[dim]  ... and {n_hits - display_limit} more[/dim]")
    if hits_path:
        console.print(f"\n[green]Wrote {n_hits} hits to {hits_path}[/green]")


def _resolve_project_query(proj) -> Path:
    """Return the project target's structure cif, fetching AF if missing."""
    from ..tui.structure import cached_structure_path, fetch_alphafold

    uniprot_id = (proj.manifest().get("target") or {}).get("uniprot_id")
    cached = cached_structure_path(uniprot_id, proj.structures_dir) \
        if uniprot_id else None
    if cached is not None:
        return cached

    if not uniprot_id:
        raise click.ClickException(
            "Project has no target UniProt id and no cached structure. "
            "Provide --query FILE.cif instead."
        )
    try:
        return fetch_alphafold(uniprot_id, proj.structures_dir)
    except Exception as e:  # noqa: BLE001 — surface any fetch failure cleanly
        raise click.ClickException(
            f"No cached structure for {uniprot_id} and AlphaFold fetch "
            f"failed ({e}). Run `beak structures {uniprot_id} "
            f"-o {proj.structures_dir}` or pass --query."
        )
