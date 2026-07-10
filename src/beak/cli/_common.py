"""Shared helpers for CLI command modules."""

import json
import click
from pathlib import Path


class JobFailed(click.ClickException):
    """A job reached a non-COMPLETED terminal state (FAILED/CANCELLED/UNKNOWN)."""
    exit_code = 1


class RemoteUnreachable(click.ClickException):
    """The remote server could not be reached (connection/timeout/OS error)."""
    exit_code = 3


def json_mode(ctx, local=False):
    """True when machine-readable output is requested.

    ORs a per-command ``--json`` flag (`local`) with the group-level
    ``--json`` carried on ``ctx.obj``, so both `beak --json search ...`
    and `beak search --json ...` work.
    """
    return bool(local or (ctx.obj or {}).get('json'))


def emit_json(payload):
    """Write one JSON object to stdout (the machine-output channel).

    ``default=str`` is a safety net for stray non-JSON types (e.g. a
    ``datetime`` from a manifest) so a payload never crashes mid-emit.
    """
    click.echo(json.dumps(payload, default=str))


# Statuses worth reusing: a matching job that is pending, running, or already
# done. FAILED/CANCELLED jobs are intentionally not reused — resubmit those.
_REUSABLE_STATES = {'SUBMITTED', 'QUEUED', 'RUNNING', 'COMPLETED'}


def job_fingerprint(job_type, input_path, params):
    """Stable content-based identity for a submission.

    Hashes the job type, the input file's *contents* (not its path, so an
    identical query under a different name still matches), and the salient
    parameters. ``input_path`` may be None (e.g. embeddings --from-job),
    in which case identity rests on ``params`` alone.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(job_type.encode())
    if input_path:
        try:
            with open(input_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
        except OSError:
            h.update(str(input_path).encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def find_reusable_job(fingerprint):
    """Return the newest non-failed job matching ``fingerprint``, or None.

    Reads ``~/.beak/jobs.json`` directly (no SSH). Used to avoid
    resubmitting an expensive job an agent already launched.
    """
    db_path = Path.home() / ".beak" / "jobs.json"
    if not db_path.exists():
        return None
    try:
        with open(db_path) as f:
            db = json.load(f)
    except (OSError, ValueError):
        return None

    matches = [
        {
            'job_id': jid,
            'job_type': info.get('job_type'),
            'name': info.get('name'),
            'status': info.get('status'),
            'submitted_at': info.get('submitted_at', ''),
        }
        for jid, info in db.items()
        if info.get('fingerprint') == fingerprint
        and info.get('status') in _REUSABLE_STATES
    ]
    if not matches:
        return None
    return max(matches, key=lambda m: m['submitted_at'])


def get_manager(job_id=None, job_type=None):
    """Create the appropriate manager for a job_id or job_type.

    Reads job_type from the local job database when job_id is given.
    Uses config defaults for connection parameters.
    """
    from ..config import get_default_connection

    defaults = get_default_connection()
    host = defaults.get('host')
    user = defaults.get('user')

    if not host or not user:
        raise click.ClickException(
            "No connection configured. Run 'beak config init' first."
        )

    if job_id and not job_type:
        db_path = Path.home() / ".beak" / "jobs.json"
        if db_path.exists():
            with open(db_path) as f:
                job_db = json.load(f)
            if job_id in job_db:
                job_type = job_db[job_id].get('job_type')

    type_map = {
        'search': 'beak.remote.search:MMseqsSearch',
        'taxonomy': 'beak.remote.taxonomy:MMseqsTaxonomy',
        'align': 'beak.remote.align:ClustalAlign',
        'embeddings': 'beak.remote.embeddings:ESMEmbeddings',
        'foldseek': 'beak.remote.foldseek:RemoteFoldseek',
        'pipeline': 'beak.remote.pipeline:Pipeline',
    }

    module_class = type_map.get(job_type, 'beak.remote.base:RemoteJobManager')
    module_path, class_name = module_class.split(':')

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    try:
        return cls()
    except (ConnectionError, TimeoutError, OSError) as e:
        raise RemoteUnreachable(f"Cannot connect to remote server: {e}")


def get_hmmer_manager():
    """Create an HmmerScan instance using config defaults."""
    from ..config import get_default_connection
    from ..remote.hmmer import HmmerScan
    from ..remote.base import RemoteJobManager
    from fabric import Connection

    defaults = get_default_connection()
    host = defaults.get('host')
    user = defaults.get('user')

    if not host or not user:
        raise click.ClickException(
            "No connection configured. Run 'beak config init' first."
        )

    key_path = defaults.get('key_path')
    if key_path is None:
        key_path = RemoteJobManager._find_ssh_key(None)

    try:
        conn = Connection(
            host=host,
            user=user,
            connect_timeout=10,
            connect_kwargs={"key_filename": str(Path(key_path).expanduser())}
        )
        return conn, HmmerScan(connection=conn)
    except (ConnectionError, TimeoutError, OSError) as e:
        raise RemoteUnreachable(f"Cannot connect to remote server: {e}")


def auto_name_from_pfam(query_file: str, job_type: str) -> str:
    """Try to generate a job name from the top Pfam hit of the query.

    Returns a name like 'Pkinase_search' on success, or a readable
    fallback like 'search_swift-folding-falcon' if Pfam is unavailable.
    """
    from ..remote.naming import generate_readable_name

    fallback = f"{job_type}_{generate_readable_name()}"

    try:
        _, hmmer = get_hmmer_manager()
        hits = hmmer.scan(query_file)
        if hits:
            return f"{hits[0]['pfam_name']}_{job_type}"
    except Exception:
        pass

    return fallback


def resolve_query_to_fasta(query: str) -> str:
    """Resolve a query argument to a local FASTA file path.

    Accepts:
        - Path to an existing FASTA file
        - UniProt accession (e.g., P0DTC2, A0A0K9RLB5)
        - Raw amino acid sequence string

    Returns:
        Path to a FASTA file (may be a temp file for string/accession inputs)
    """
    import re
    import tempfile

    if Path(query).exists():
        return query

    # UniProt accession pattern: 6-10 alphanumeric, starts with letter
    if re.fullmatch(r'[A-Z][A-Z0-9]{4,9}', query, re.IGNORECASE):
        from ..api.uniprot import fetch_uniprot
        try:
            return fetch_uniprot(query)
        except ValueError:
            pass

    cleaned = re.sub(r'\s+', '', query)
    if re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwyx*-]+', cleaned):
        tmp = Path(tempfile.mkdtemp()) / "query.fasta"
        tmp.write_text(f">query\n{cleaned}\n")
        return str(tmp)

    raise click.BadParameter(
        f"'{query}' is not a FASTA file, UniProt accession, or valid protein sequence.",
        param_hint="'QUERY'",
    )


def get_remote_file_age(conn, remote_path: str) -> str:
    """Get human-readable age of a remote file via stat."""
    import datetime

    result = conn.run(
        f'stat -c %Y {remote_path} 2>/dev/null || stat -f %m {remote_path} 2>/dev/null',
        hide=True, warn=True,
    )
    if not result.ok or not result.stdout.strip():
        return "unknown"

    try:
        mtime = int(result.stdout.strip())
        age_days = int((datetime.datetime.now().timestamp() - mtime) / 86400)
        mod_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

        if age_days == 0:
            return f"today ({mod_date})"
        elif age_days == 1:
            return f"1 day ({mod_date})"
        else:
            age_str = f"{age_days} days ({mod_date})"
            if age_days > 180:
                age_str += " [yellow](consider updating)[/yellow]"
            return age_str
    except (ValueError, OSError):
        return "unknown"
