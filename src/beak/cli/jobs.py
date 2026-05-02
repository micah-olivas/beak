"""Job management commands: jobs, status, log, cancel, results."""

import json
import click
from pathlib import Path

from .main import main
from ._common import get_manager


@main.command()
@click.option('--type', 'job_type', default=None,
              type=click.Choice(['search', 'taxonomy', 'align', 'embeddings', 'pipeline']),
              help='Filter by job type')
@click.option('--status', 'status_filter', default=None,
              help='Filter by status (RUNNING, COMPLETED, FAILED, etc.)')
@click.option('--no-refresh', is_flag=True,
              help='Skip the automatic remote refresh of non-terminal jobs')
def jobs(job_type, status_filter, no_refresh):
    """List all jobs.

    By default, connects to the remote server once to refresh the status of
    any SUBMITTED or RUNNING jobs. Terminal states (COMPLETED, FAILED,
    CANCELLED) are read from the local cache without network calls. Pass
    --no-refresh to skip the refresh entirely (e.g. when offline).
    """
    db_path = Path.home() / ".beak" / "jobs.json"
    if not db_path.exists():
        click.echo("No jobs found.")
        return

    with open(db_path) as f:
        job_db = json.load(f)

    if not job_db:
        click.echo("No jobs found.")
        return

    refreshed_count = 0
    refresh_error = None
    if not no_refresh:
        active_ids = [
            jid for jid, info in job_db.items()
            if info.get('status', 'UNKNOWN') not in ('COMPLETED', 'FAILED', 'CANCELLED')
        ]
        if active_ids:
            try:
                # Any RemoteJobManager subclass reads status.txt the same way,
                # so use a lightweight one (search) to avoid ESMEmbeddings'
                # Docker preflight on every `beak jobs` call.
                mgr = get_manager(job_type='search')
                for jid in active_ids:
                    old = job_db[jid].get('status', 'UNKNOWN')
                    result = mgr.status(jid)
                    if result['status'] != old:
                        refreshed_count += 1
                # Reload job_db after status() calls updated it
                with open(db_path) as f:
                    job_db = json.load(f)
            except Exception as e:
                refresh_error = str(e)

    if refreshed_count:
        click.echo(f"Refreshed {refreshed_count} job(s)\n")
    elif refresh_error:
        click.echo(f"(Could not refresh: {refresh_error}; showing cached state)\n")

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

    name_width = 20
    header = f"{'ID':<10} {'Name':<{name_width}} {'Type':<12} {'Status':<12} {'Submitted'}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        name = r['name']
        if len(name) > name_width:
            name = name[:name_width - 1] + '…'
        click.echo(f"{r['id']:<10} {name:<{name_width}} {r['type']:<12} {r['status']:<12} {r['submitted']}")

    non_terminal = [r for r in rows if r['status'] in ('SUBMITTED', 'RUNNING')]
    if non_terminal:
        click.echo(
            "\nTip: `beak status <ID> --watch` for a live view of a running job."
        )


@main.command()
@click.argument('job_id')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress')
@click.option('--watch', '-w', is_flag=True, help='Live-updating status display')
@click.option('--interval', default=2.0, help='Refresh interval in seconds (with --watch)')
def status(job_id, verbose, watch, interval):
    """Check job status"""
    from .display import print_status, watch_status

    mgr = get_manager(job_id=job_id)

    if watch:
        watch_status(mgr, job_id, interval=interval)
        return

    info = mgr.detailed_status(job_id)
    print_status(info)


@main.command()
@click.argument('job_id')
@click.option('--lines', '-n', default=50, help='Number of log lines')
@click.option('--follow', '-f', is_flag=True,
              help='Stream new log lines live until the job finishes (Ctrl-C to stop)')
@click.option('--interval', default=2.0,
              help='Poll interval in seconds (with --follow)')
def log(job_id, lines, follow, interval):
    """View job log.

    Without --follow, prints the last N lines and a directory listing.
    With --follow, tails the log live: shows the last 20 lines for
    context and then streams new output as the job runs, exiting once
    the job reaches a terminal state.
    """
    mgr = get_manager(job_id=job_id)
    if follow:
        mgr.follow_log(job_id, interval=interval)
    else:
        mgr.get_log(job_id, lines=lines)


@main.command()
@click.argument('job_id')
def cancel(job_id):
    """Cancel a running job"""
    mgr = get_manager(job_id=job_id)
    mgr.cancel(job_id)


@main.command()
@click.argument('job_id')
@click.option('--parse', is_flag=True, help='Parse results and print summary')
@click.option('--taxonomy', '-t', 'with_taxonomy', is_flag=True,
              help='(search jobs) Build hits_taxonomy.tsv on the remote via '
                   '`mmseqs convertalis` if it is missing, then download. '
                   'Use this for searches submitted before taxonomy-by-default.')
def results(job_id, parse, with_taxonomy):
    """Download job results."""
    mgr = get_manager(job_id=job_id)

    # Embeddings have a richer results story than "print the DataFrame" —
    # they produce a pickle the user loads in Python, so show a summary
    # and a ready-to-paste loader snippet instead of raw values.
    if mgr.JOB_TYPE == 'embeddings':
        _show_embeddings_results(mgr, job_id)
        return

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

    # Extract / fetch taxonomy for search jobs. Default behavior is
    # "download if already on the remote"; --taxonomy forces the build
    # when missing (useful for pre-taxonomy-by-default searches).
    if mgr.JOB_TYPE == 'search':
        _maybe_show_search_taxonomy(mgr, job_id, force_build=with_taxonomy)


def _maybe_show_search_taxonomy(mgr, job_id: str, force_build: bool = False):
    """Download (and optionally build) hits_taxonomy.tsv for a search job.

    When ``force_build`` is False, only pulls the TSV if it already
    exists remotely — the default `beak results` call shouldn't trigger
    a convertalis rebuild on every invocation. When True, runs
    ensure_remote_hits_taxonomy first, so the TSV is materialized even
    for searches submitted before taxonomy-by-default.
    """
    try:
        if force_build:
            tax = mgr.get_hit_taxonomy(job_id, refresh=False)
        else:
            # Peek at the remote: only download if it's already there.
            job_db = mgr._load_job_db()
            remote_path = job_db[job_id]['remote_path']
            remote_tsv = f"{remote_path}/hits_taxonomy.tsv"
            check = mgr.conn.run(
                f'[ -s {remote_tsv} ] && echo OK || echo MISSING',
                hide=True, warn=True,
            )
            if check.stdout.strip() != 'OK':
                return  # silent — searches without taxonomy shouldn't be noisy
            tax = mgr.get_hit_taxonomy(job_id)
    except Exception as exc:  # noqa: BLE001 — taxonomy is opportunistic
        click.echo(f"(couldn't fetch taxonomy: {exc})")
        return

    if tax.empty:
        if force_build:
            click.echo(
                "(target DB has no taxonomy — no hits_taxonomy.tsv produced)"
            )
        return

    from pathlib import Path
    project_dir = mgr.get_project_dir(job_id)
    tsv_path = Path(project_dir) / "hits_taxonomy.tsv"
    n_domains = tax['domain'].nunique(dropna=True)
    n_phyla = tax['phylum'].nunique(dropna=True)
    click.echo(
        f"Taxonomy: {len(tax):,} hits, {n_domains} domain(s), "
        f"{n_phyla} phyla  →  {tsv_path}"
    )
    click.echo("  from beak.embeddings import load_hit_taxonomy")
    click.echo(f"  tax = load_hit_taxonomy('{tsv_path}')")


def _show_embeddings_results(mgr, job_id: str):
    """Download an embeddings job and print a human-friendly summary."""
    embeddings_dir = mgr.download(job_id)

    from ..embeddings import (
        load_mean_embeddings,
        load_per_token_embeddings,
        load_hit_taxonomy,
    )

    info = mgr._load_job_db().get(job_id, {})
    model = info.get('model', '?')
    params = info.get('parameters', {}) or {}

    mean_path = embeddings_dir / 'mean_embeddings.pkl'
    tok_path = embeddings_dir / 'per_token_embeddings.pkl'
    failed_path = embeddings_dir / 'failed.tsv'
    tax_path = embeddings_dir / 'hits_taxonomy.tsv'

    click.echo(f"\nModel: {model}")
    click.echo(f"Layers: {params.get('repr_layers', [])}")

    if mean_path.exists():
        df = load_mean_embeddings(mean_path)
        click.echo(
            f"Mean embeddings:       {df.shape[0]:,} sequences × "
            f"{df.shape[1]:,} dimensions  →  {mean_path}"
        )

    if tok_path.exists():
        df = load_per_token_embeddings(tok_path)
        click.echo(
            f"Per-token embeddings:  {len(df.index.get_level_values('seq_id').unique()):,} sequences, "
            f"{df.shape[0]:,} total residues × "
            f"{df.shape[1]:,} dimensions  →  {tok_path}"
        )

    if failed_path.exists():
        n_failed = sum(1 for _ in open(failed_path))
        if n_failed:
            click.echo(f"Failed sequences:      {n_failed}  →  {failed_path}")

    tax_df = None
    if tax_path.exists() and tax_path.stat().st_size > 0:
        tax_df = load_hit_taxonomy(tax_path)
        n_domains = tax_df['domain'].nunique(dropna=True) if not tax_df.empty else 0
        n_phyla = tax_df['phylum'].nunique(dropna=True) if not tax_df.empty else 0
        click.echo(
            f"Taxonomy:              {len(tax_df):,} hits, "
            f"{n_domains} domain(s), {n_phyla} phyla  →  {tax_path}"
        )

    click.echo("\nLoad in Python:")
    if mean_path.exists():
        click.echo("  from beak.embeddings import load_mean_embeddings")
        click.echo(f"  df = load_mean_embeddings('{mean_path}')")
    if tok_path.exists():
        click.echo("  from beak.embeddings import load_per_token_embeddings")
        click.echo(f"  tok = load_per_token_embeddings('{tok_path}')")
    if tax_df is not None:
        click.echo("  from beak.embeddings import load_hit_taxonomy")
        click.echo(f"  tax = load_hit_taxonomy('{tax_path}')")
        click.echo("  # plot_pca(df, color=tax.loc[df.index, 'domain'])")
