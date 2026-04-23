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
def log(job_id, lines):
    """View job log"""
    mgr = get_manager(job_id=job_id)
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
def results(job_id, parse):
    """Download job results"""
    mgr = get_manager(job_id=job_id)

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
