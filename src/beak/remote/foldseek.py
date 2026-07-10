"""Remote foldseek structural-search job manager.

Runs `foldseek easy-search` on the remote server against a database that
lives on the remote — the same submit → poll → pull lifecycle as
``MMseqsSearch``. Foldseek is structural MMseqs2 (same authors, same
`easy-search`/`databases` CLI shape), so this mirrors ``remote/search.py``
closely; the query is a structure file (mmCIF/PDB) rather than a FASTA and
the remote command is a single `easy-search` (no separate `createdb`).
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union

import pandas as pd

from .base import RemoteJobManager
from ..structures.foldseek import DEFAULT_OUTPUT_COLUMNS, parse_foldseek_m8


# Friendly alias -> (foldseek `databases` download name, subdir under a DB
# root). `beak setup foldseek --db <alias>` downloads the database and
# `AVAILABLE_DBS` (below) lets `--db <alias>` resolve to the shared install
# under DB_BASE_PATH. Non-shared installs are addressed by their configured
# absolute prefix instead (see the `foldseek` CLI command).
FOLDSEEK_DATABASES = {
    'pdb':            ('PDB',                 'foldseek/pdb'),
    'afdb_swissprot': ('Alphafold/Swiss-Prot', 'foldseek/afdb_swissprot'),
    'afdb_proteome':  ('Alphafold/Proteome',   'foldseek/afdb_proteome'),
    'afdb50':         ('Alphafold/UniProt50',  'foldseek/afdb50'),
    'afdb_uniprot':   ('Alphafold/UniProt',    'foldseek/afdb_uniprot'),
    'esmatlas30':     ('ESMAtlas30',           'foldseek/esmatlas30'),
    'cath50':         ('CATH50',               'foldseek/cath50'),
    'bfvd':           ('BFVD',                 'foldseek/bfvd'),
}


class RemoteFoldseek(RemoteJobManager):
    """Foldseek structural-search job manager for remote execution."""

    JOB_TYPE = 'foldseek'
    LOG_FILE = 'foldseek.log'
    # Best-effort progress keywords, ordered by foldseek's pipeline. Missing
    # matches just leave a plain spinner, so an imperfect list is harmless.
    LOG_OPERATIONS = [
        ('createdb',    'Preparing query'),
        ('Prefilter',   'Prefiltering'),
        ('Alignment',   'Structural alignment'),
        ('convertalis', 'Formatting results'),
    ]

    # alias -> database prefix relative to DB_BASE_PATH, for the shared
    # install layout `beak setup foldseek --system` writes.
    AVAILABLE_DBS = {
        alias: f"{subdir}/db" for alias, (_, subdir) in FOLDSEEK_DATABASES.items()
    }

    @staticmethod
    def _format_params(params: Dict) -> str:
        """Render a param dict as a foldseek CLI flag string.

        Single-char keys get ``-`` (e.g. ``-s``); longer keys get ``--`` and
        underscores become dashes (``max_seqs`` -> ``--max-seqs``). Matches
        the convention in ``MMseqsSearch.submit``.
        """
        parts = []
        for k, v in params.items():
            name = k.replace("_", "-")
            prefix = "-" if len(k) == 1 else "--"
            parts.append(f'{prefix}{name} {v}')
        return ' '.join(parts)

    @staticmethod
    def _build_job_script(remote_job_path: str, remote_query: str,
                          db_path: str, param_str: str) -> str:
        """Build the remote bash script for one easy-search job.

        Pure string builder (no SSH) so it is unit-testable. Uses
        ``pipefail`` + an ``if`` so a foldseek failure — not tee's exit —
        drives the FAILED marker, and always writes a terminal status line
        so the status machinery classifies the job correctly.
        """
        cols = ",".join(DEFAULT_OUTPUT_COLUMNS)
        return f"""#!/bin/bash
set -o pipefail

# Job metadata
echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

# Run foldseek structural search (reads the query structure directly)
if foldseek easy-search \\
  {remote_query} \\
  {db_path} \\
  {remote_job_path}/results.m8 \\
  {remote_job_path}/tmp \\
  --format-output "{cols}" \\
  {param_str} \\
  2>&1 | tee {remote_job_path}/foldseek.log; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi

rm -rf {remote_job_path}/tmp
"""

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        return {'database': info.get('database', '')}

    def list_databases(self) -> pd.DataFrame:
        """List known foldseek databases and whether they exist on the remote."""
        db_info = []
        for alias, db_rel in self.AVAILABLE_DBS.items():
            db_path = f"{self.DB_BASE_PATH}/{db_rel}"
            # Foldseek DBs are prefix-based; the `.dbtype` sidecar is the
            # reliable existence marker. du -sh the containing directory.
            result = self.conn.run(
                f'if [ -e {db_path}.dbtype ]; then '
                f'du -sh $(dirname {db_path}) 2>/dev/null | cut -f1; '
                f'else echo "NOT_FOUND"; fi',
                hide=True, warn=True,
            )
            size = result.stdout.strip()
            exists = size != "NOT_FOUND"
            db_info.append({
                'alias': alias,
                'database_name': FOLDSEEK_DATABASES[alias][0],
                'path': db_path,
                'exists': exists,
                'size': size if exists else 'N/A',
            })
        return pd.DataFrame(db_info).sort_values('alias')

    def submit(self,
               query_file: str,
               database: str,
               job_name: Optional[str] = None,
               quiet: bool = False,
               **foldseek_params) -> str:
        """Submit a foldseek structural-search job.

        ``query_file`` is a local structure file (mmCIF/PDB); ``database``
        is an alias in :data:`AVAILABLE_DBS` or an absolute remote prefix.
        ``foldseek_params`` map to easy-search flags (e.g. ``s=9.5``,
        ``e=0.001``, ``max_seqs=1000``).
        """
        job_id = str(uuid.uuid4())[:8]

        self.create_project(job_id=job_id, job_type='foldseek',
                            name=job_name, query_file=query_file)

        if not job_name:
            from .naming import generate_readable_name
            job_name = f"foldseek_{generate_readable_name()}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        final_params = dict(foldseek_params)
        # Thread cap so a query doesn't monopolize the shared server; an
        # explicit override still wins.
        from ..config import get_compute_threads
        final_params.setdefault('threads', get_compute_threads())

        db_path = self._resolve_database(database)
        db_check = self.conn.run(
            f'[ -e {db_path}.dbtype ] || [ -e {db_path} ] && echo EXISTS || echo NOT_FOUND',
            hide=True, warn=True,
        )
        if db_check.stdout.strip() == "NOT_FOUND":
            available = ", ".join(self.AVAILABLE_DBS.keys())
            raise ValueError(
                f"Foldseek database '{database}' not found at {db_path}\n"
                f"Available aliases: {available}\n"
                f"Or run `beak setup foldseek --db <name>` to download one, "
                f"or pass a full remote path."
            )

        self.conn.run(f'mkdir -p {remote_job_path}/tmp', hide=True)

        # Stage the query, preserving its extension so foldseek infers the
        # structure format (.cif/.pdb/.mmcif).
        suffix = Path(query_file).suffix or '.pdb'
        remote_query = f"{remote_job_path}/query{suffix}"
        self.conn.put(query_file, remote_query)

        param_str = self._format_params(final_params)
        job_script = self._build_job_script(
            remote_job_path, remote_query, db_path, param_str)

        script_path = f"{remote_job_path}/run.sh"
        self.conn.put(local=self._write_temp_script(job_script),
                      remote=script_path)
        self.conn.run(f'chmod +x {script_path}', hide=True)

        result = self.conn.run(
            f'nohup {script_path} > {remote_job_path}/nohup.out 2>&1 & echo $!',
            hide=True,
        )
        pid = result.stdout.strip()
        self.conn.run(f'echo {pid} > {remote_job_path}/pid.txt', hide=True)

        with self._mutate_job_db() as db:
            db[job_id] = {
                'job_type': 'foldseek',
                'name': job_name,
                'database': database,
                'database_path': db_path,
                'input_file': str(query_file),
                'remote_path': remote_job_path,
                'submitted_at': datetime.now().isoformat(),
                'status': 'SUBMITTED',
                'pid': pid,
                'parameters': final_params,
            }

        if not quiet:
            print(f"✓ Submitted {job_name} → {database} ({job_id})")

        return job_id

    def get_results(self, job_id: str,
                    parse: bool = True) -> Union[Path, pd.DataFrame]:
        """Download (and optionally parse) foldseek hits for a completed job.

        Returns a DataFrame of structural hits when ``parse`` is True, else
        the path to the downloaded ``results.m8``.
        """
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")

        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            project_dir = self.create_project(job_id, 'foldseek')

        remote_path = self._load_job_db()[job_id]['remote_path']

        m8_file = project_dir / "results.m8"
        if not m8_file.exists():
            self.conn.get(f"{remote_path}/results.m8", str(m8_file))

        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(f"{remote_path}/foldseek.log", str(log_file))

        if parse:
            return parse_foldseek_m8(m8_file.read_text(), DEFAULT_OUTPUT_COLUMNS)
        return m8_file
