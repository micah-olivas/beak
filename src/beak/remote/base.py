"""Base class for remote job management with shared lifecycle methods"""

import re
import json
import os
import threading
import time
import shutil
import pandas as pd
from pathlib import Path

from contextlib import contextmanager
from fabric import Connection
from datetime import datetime
from typing import Dict, Optional


# Per-process lock guarding `~/.beak/jobs.json`. Multiple worker threads
# (status polls, submits, pulls) all read-modify-write this file; without
# serialization two writers could clobber each other's updates. Module-
# level + RLock so an outer mutate-style helper can call the same db
# inside the same thread without deadlocking.
_JOB_DB_LOCK = threading.RLock()


class RemoteJobManager:
    """Base class for remote job management.

    Subclasses should define:
        JOB_TYPE: str - job type identifier (e.g., 'search', 'taxonomy')
        LOG_FILE: str - name of the log file (e.g., 'mmseqs.log')
        LOG_OPERATIONS: list - keyword/label pairs for progress parsing
        AVAILABLE_DBS: dict - database alias -> filename mapping (optional)
    """

    # Subclass overrides
    JOB_TYPE = 'unknown'
    LOG_FILE = 'job.log'
    LOG_OPERATIONS = []
    AVAILABLE_DBS = {}

    LOCAL_PROJECTS_DIR = Path.home() / "beak_projects"
    DB_BASE_PATH = "/srv/protein_sequence_databases"

    def __init__(self, host: Optional[str] = None, user: Optional[str] = None,
                 key_path: Optional[str] = None, remote_job_dir: Optional[str] = None,
                 connection: Optional[Connection] = None):
        if connection is not None:
            self.conn = connection
        else:
            # Fall back to config defaults for missing parameters
            from ..config import get_default_connection
            defaults = get_default_connection()
            host = host or defaults.get('host')
            user = user or defaults.get('user')
            key_path = key_path or defaults.get('key_path')
            remote_job_dir = remote_job_dir or defaults.get('remote_job_dir')

            if not host or not user:
                raise ValueError(
                    "host and user are required. Provide them as arguments or "
                    "configure defaults with: beak config init"
                )

            if key_path is None:
                key_path = self._find_ssh_key()

            from .session import _no_stdin_config
            self.conn = Connection(
                host=host,
                user=user,
                connect_timeout=10,
                connect_kwargs={"key_filename": str(Path(key_path).expanduser())},
                config=_no_stdin_config(),
            )

        # Default remote job directory
        if remote_job_dir is None:
            remote_job_dir = "~/beak_jobs"

        # Expand tilde in remote path for SFTP compatibility
        if remote_job_dir.startswith('~'):
            try:
                result = self.conn.run('echo $HOME', hide=True)
            except Exception as e:
                raise ConnectionError(
                    f"Could not connect to {self.conn.host}: {e}"
                ) from e
            remote_job_dir = result.stdout.strip() + remote_job_dir[1:]

        self.remote_job_dir = remote_job_dir
        self.local_job_db = Path.home() / ".beak" / "jobs.json"

        # Setup
        self._setup_remote_dir()
        self._setup_local_db()
        self._setup_projects_dir()

    # ── SSH key detection ───────────────────────────────────────────

    def _find_ssh_key(self) -> str:
        """Auto-detect SSH key from common locations"""
        ssh_dir = Path.home() / ".ssh"
        key_names = ['id_ed25519', 'id_rsa', 'id_ecdsa']

        for key_name in key_names:
            key_path = ssh_dir / key_name
            if key_path.exists():
                return str(key_path)

        raise FileNotFoundError(
            f"No SSH key found in {ssh_dir}. "
            f"Please generate one with: ssh-keygen -t ed25519 "
            f"or specify key_path explicitly."
        )

    # ── Setup helpers ───────────────────────────────────────────────

    def _setup_remote_dir(self):
        """Ensure remote job directory exists"""
        self.conn.run(f'mkdir -p {self.remote_job_dir}', hide=True)

    def _setup_local_db(self):
        """Setup local job tracking database"""
        self.local_job_db.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_job_db.exists():
            self._save_job_db({})

    def _setup_projects_dir(self):
        """Setup local projects directory with index"""
        self.LOCAL_PROJECTS_DIR.mkdir(exist_ok=True)
        index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
        if not index_file.exists():
            index_file.write_text(json.dumps({}, indent=2))

    # ── Remote environment verification ───────────────────────────

    # Tools grouped by what needs them
    REQUIRED_TOOLS = {
        'mmseqs':  {'needed_by': 'search, taxonomy', 'install': 'https://github.com/soedinglab/MMseqs2'},
        'python3': {'needed_by': 'pipelines (filtering)', 'install': 'apt install python3'},
    }
    OPTIONAL_TOOLS = {
        'clustalo': {'needed_by': 'align (clustalo)', 'install': 'http://www.clustal.org/omega/'},
        'mafft':    {'needed_by': 'align (mafft)', 'install': 'https://mafft.cbrc.jp/alignment/software/'},
        'muscle':   {'needed_by': 'align (muscle)', 'install': 'https://github.com/rcedgar/muscle'},
        'iqtree2':  {'needed_by': 'tree', 'install': 'http://www.iqtree.org/'},
        'iqtree':   {'needed_by': 'tree (fallback)', 'install': 'http://www.iqtree.org/'},
        'docker':   {'needed_by': 'embeddings', 'install': 'https://docs.docker.com/engine/install/'},
        'seqkit':   {'needed_by': 'sequence filtering (optional)', 'install': 'https://bioinf.shenwei.me/seqkit/'},
        'hmmscan':  {'needed_by': 'pfam (domain search)', 'install': 'http://hmmer.org/'},
        'hmmpress': {'needed_by': 'pfam (database setup)', 'install': 'http://hmmer.org/'},
    }

    def verify_remote(self, verbose: bool = True) -> Dict:
        """
        Check which tools and databases are available on the remote server.

        Args:
            verbose: Print results as they're checked

        Returns:
            Dict with 'tools' and 'databases' status information
        """
        results = {'tools': {}, 'databases': {}, 'ok': True}

        all_tools = {**self.REQUIRED_TOOLS, **self.OPTIONAL_TOOLS}

        if verbose:
            print(f"Checking remote environment on {self.conn.host}...")
            print()

        # Check tools
        for tool, info in all_tools.items():
            required = tool in self.REQUIRED_TOOLS
            check = self.conn.run(
                f'command -v {tool} 2>/dev/null && {tool} --version 2>&1 | head -1 || echo "__NOT_FOUND__"',
                hide=True, warn=True
            )
            output = check.stdout.strip()
            found = '__NOT_FOUND__' not in output

            # Some tools don't support --version; just check command -v
            if not found:
                check2 = self.conn.run(f'command -v {tool}', hide=True, warn=True)
                found = check2.ok
                output = check2.stdout.strip() if found else ''

            results['tools'][tool] = {
                'found': found,
                'required': required,
                'version': output.split('\n')[-1] if found else None,
                'needed_by': info['needed_by'],
                'install': info['install'],
            }

            if verbose:
                tag = 'required' if required else 'optional'
                if found:
                    ver = results['tools'][tool]['version'] or ''
                    if ver:
                        ver = f" ({ver})"
                    print(f"  ✓ {tool}{ver}")
                else:
                    marker = '✗' if required else '○'
                    print(f"  {marker} {tool} — not found [{tag}, {info['needed_by']}]")

            if required and not found:
                results['ok'] = False

        # Check database directory
        if verbose:
            print()
        db_check = self.conn.run(
            f'[ -d {self.DB_BASE_PATH} ] && echo "EXISTS" || echo "NOT_FOUND"',
            hide=True, warn=True
        )
        db_dir_exists = db_check.stdout.strip() == 'EXISTS'

        if db_dir_exists:
            ls_result = self.conn.run(
                f'ls {self.DB_BASE_PATH}/ 2>/dev/null | head -20',
                hide=True, warn=True
            )
            db_names = [d for d in ls_result.stdout.strip().split('\n') if d]
            results['databases'] = {
                'path': self.DB_BASE_PATH,
                'exists': True,
                'count': len(db_names),
                'names': db_names,
            }
            if verbose:
                print(f"  ✓ Database directory: {self.DB_BASE_PATH} ({len(db_names)} databases)")
        else:
            results['databases'] = {
                'path': self.DB_BASE_PATH,
                'exists': False,
                'count': 0,
                'names': [],
            }
            results['ok'] = False
            if verbose:
                print(f"  ✗ Database directory not found: {self.DB_BASE_PATH}")

        # Check disk space
        disk_result = self.conn.run(
            f'df -h {self.remote_job_dir} 2>/dev/null | tail -1',
            hide=True, warn=True
        )
        if disk_result.ok and disk_result.stdout.strip():
            parts = disk_result.stdout.strip().split()
            if len(parts) >= 4:
                results['disk'] = {
                    'total': parts[1],
                    'used': parts[2],
                    'available': parts[3],
                }
                if verbose:
                    print(f"  ✓ Disk space: {parts[3]} available")

        if verbose:
            print()
            if results['ok']:
                print("All required tools found.")
            else:
                missing = [t for t, s in results['tools'].items()
                           if s['required'] and not s['found']]
                if missing:
                    print(f"Missing required tools: {', '.join(missing)}")
                    print("Install them on the remote server before submitting jobs.")

        return results

    # ── Job database ────────────────────────────────────────────────

    def _load_job_db(self) -> Dict:
        """Load local job database, tolerating a partially-written file.

        A SIGKILL or disk-full mid-`_save_job_db` could leave the file
        truncated; `json.JSONDecodeError` is treated as "no jobs yet"
        rather than crashing the whole TUI. Holds the lock so a
        concurrent writer can't move the file out from under us.
        """
        with _JOB_DB_LOCK:
            try:
                with open(self.local_job_db, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}

    def _save_job_db(self, db: Dict):
        """Save local job database via temp + atomic rename.

        Without this, two workers writing the file simultaneously could
        interleave their writes (last-writer-wins, lost updates), and a
        crash mid-write would corrupt jobs.json. The lock + tmp+replace
        pattern eliminates both classes of bug — readers always see a
        complete file, writers never lose updates.
        """
        with _JOB_DB_LOCK:
            tmp = self.local_job_db.with_suffix(self.local_job_db.suffix + '.tmp')
            with open(tmp, 'w') as f:
                json.dump(db, f, indent=2)
                # fsync the data to disk before the rename so a power
                # loss can't leave a renamed-but-zero-byte file.
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, self.local_job_db)

    @contextmanager
    def _mutate_job_db(self):
        """Atomic read-modify-write of the job database.

            with self._mutate_job_db() as db:
                db[job_id]['status'] = 'COMPLETED'

        Holds `_JOB_DB_LOCK` for the whole block so no other writer can
        race a stale read with a fresh write.
        """
        with _JOB_DB_LOCK:
            db = self._load_job_db()
            yield db
            self._save_job_db(db)

    def _write_temp_script(self, content: str) -> str:
        """Write script to temporary file and return path"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh')
        tmp.write(content)
        tmp.close()
        return tmp.name

    # ── Project management ──────────────────────────────────────────

    def _get_projects_index(self) -> Dict:
        """Load projects index"""
        index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
        if not index_file.exists():
            return {}
        with open(index_file) as f:
            return json.load(f)

    def _update_projects_index(self, job_id: str, project_dir: Path,
                            job_type: str, name: str):
        """Add/update project in index"""
        index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
        index = self._get_projects_index()

        index[job_id] = {
            'project_dir': str(project_dir),
            'job_type': job_type,
            'name': name,
            'created': datetime.now().isoformat()
        }

        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def create_project(self, job_id: str, job_type: str,
                    name: Optional[str] = None,
                    query_file: Optional[str] = None) -> Path:
        """Create local project directory"""
        if name:
            safe_name = "".join(
                c if c.isalnum() or c in '-_' else '_'
                for c in name.lower().replace(' ', '_')
            )
            project_name = f"{safe_name}_{job_id}"
        else:
            project_name = f"{job_type}_{job_id}"

        self.LOCAL_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

        project_dir = self.LOCAL_PROJECTS_DIR / project_name
        project_dir.mkdir(exist_ok=True)

        metadata = {
            'job_id': job_id,
            'job_type': job_type,
            'name': name or project_name,
            'created': datetime.now().isoformat()
        }

        with open(project_dir / ".metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if query_file:
            shutil.copy2(query_file, project_dir / "input.fasta")

        self._update_projects_index(job_id, project_dir, job_type, name or project_name)
        return project_dir

    def get_project_dir(self, job_id: str) -> Optional[Path]:
        """Get project directory for a job ID"""
        index = self._get_projects_index()
        if job_id in index:
            return Path(index[job_id]['project_dir'])
        return None

    def list_projects(self) -> pd.DataFrame:
        """List all local projects"""
        index = self._get_projects_index()

        if not index:
            return pd.DataFrame(columns=['job_id', 'name', 'type', 'created', 'size_mb'])

        projects = []
        for job_id, info in index.items():
            project_dir = Path(info['project_dir'])

            if project_dir.exists():
                size = sum(f.stat().st_size for f in project_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
            else:
                size_mb = 0

            projects.append({
                'job_id': job_id,
                'name': info['name'],
                'type': info['job_type'],
                'created': info['created'],
                'size_mb': f"{size_mb:.1f}"
            })

        df = pd.DataFrame(projects)
        return df.sort_values('created', ascending=False)

    def delete_project(self, job_id: str, confirm: bool = True):
        """Delete a local project"""
        project_dir = self.get_project_dir(job_id)

        if not project_dir:
            raise ValueError(f"Project {job_id} not found")

        if confirm:
            response = input(f"Delete project {project_dir.name}? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled")
                return

        if project_dir.exists():
            shutil.rmtree(project_dir)

        index = self._get_projects_index()
        if job_id in index:
            del index[job_id]
            index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)

        job_db = self._load_job_db()
        if job_id in job_db:
            del job_db[job_id]
            self._save_job_db(job_db)

        print(f"✓ Deleted project {job_id}")

    # ── Database resolution (shared by search/taxonomy) ─────────────

    def _resolve_database(self, database: str) -> str:
        """Resolve database alias to full path"""
        if database in self.AVAILABLE_DBS:
            db_file = self.AVAILABLE_DBS[database]
            return f"{self.DB_BASE_PATH}/{db_file}"

        if database.startswith('/'):
            return database

        return f"{self.DB_BASE_PATH}/{database}"

    # ── Status checking ─────────────────────────────────────────────

    # Seconds after submission with no status.txt and no live PID before we
    # assume the submission script died before it could write anything.
    _LAUNCH_GRACE_SECONDS = 60

    def _infer_status(self, job_info: Dict, is_running: bool,
                      status_lines: list) -> str:
        """Classify a job based on remote state.

        Order matters here. The submit script appends to status.txt
        (`echo X >> status.txt`), so a finished job's file looks like
        `Job started: ... \n RUNNING \n COMPLETED`. We have to check
        terminal markers (COMPLETED / FAILED / CANCELLED) first, since
        the literal string `RUNNING` is *always* present once the
        script started.

        The other subtle case: a job whose script was kill -9'd, OOM'd,
        or interrupted by a server reboot leaves status.txt at
        `... RUNNING` forever — no terminal marker ever gets written.
        Combined with a dead PID, that's a crashed job, not a live one.
        We surface it as FAILED past the launch grace window so the UI
        offers a Clear path instead of pretending it's still in flight.
        """
        # Terminal states win unambiguously.
        if 'COMPLETED' in status_lines:
            return 'COMPLETED'
        if 'FAILED' in status_lines:
            return 'FAILED'
        if 'CANCELLED' in status_lines:
            return 'CANCELLED'

        # Live PID → genuinely running, regardless of what status.txt says.
        if is_running:
            return 'RUNNING'

        # PID dead, no terminal marker. Two sub-cases handled together:
        # (a) status.txt has RUNNING but no terminal marker → script
        #     crashed mid-run.
        # (b) status.txt is missing entirely → submit script died
        #     before it could write anything.
        # In both cases, past the launch grace window we surface FAILED
        # so the user can clear the lingering job. Within the grace
        # window we keep it as SUBMITTED — the script may still be
        # spinning up.
        submitted_at_str = job_info.get('submitted_at')
        if submitted_at_str:
            try:
                submitted_at = datetime.fromisoformat(submitted_at_str)
                age = (datetime.now() - submitted_at).total_seconds()
                if age > self._LAUNCH_GRACE_SECONDS:
                    return 'FAILED'
            except ValueError:
                pass
        return 'SUBMITTED'

    def status(self, job_id: str, verbose: bool = False) -> Dict:
        """
        Check job status

        Args:
            job_id: Job identifier
            verbose: If True, return detailed status with progress
        """
        if verbose:
            return self.detailed_status(job_id)

        job_db = self._load_job_db()

        if job_id not in job_db:
            return {'status': 'UNKNOWN', 'error': 'Job ID not found'}

        job_info = job_db[job_id]
        remote_path = job_info['remote_path']

        # Check if process is still running
        pid = job_info['pid']
        ps_result = self.conn.run(f'ps -p {pid} -o pid=', warn=True, hide=True)
        is_running = ps_result.ok

        # Read status file
        status_result = self.conn.run(
            f'cat {remote_path}/status.txt 2>/dev/null || echo "NO_STATUS"',
            hide=True, warn=True
        )
        status_lines = status_result.stdout.strip().split('\n')

        status = self._infer_status(job_info, is_running, status_lines)

        # Calculate runtime
        submitted_at = datetime.fromisoformat(job_info['submitted_at'])
        runtime = str(datetime.now() - submitted_at).split('.')[0]

        # Update local DB. Lock held only over the in-memory mutation
        # — not over the SSH round-trip above — so concurrent polls
        # don't serialize on each other's network latency.
        with self._mutate_job_db() as db:
            entry = db.get(job_id)
            if entry is not None:
                entry['status'] = status
                entry['last_checked'] = datetime.now().isoformat()

        return {
            'job_id': job_id,
            'name': job_info['name'],
            'status': status,
            'runtime': runtime,
            'job_type': job_info.get('job_type', self.JOB_TYPE)
        }

    def detailed_status(self, job_id: str) -> Dict:
        """Get detailed status with progress parsing and per-stage info"""
        job_db = self._load_job_db()

        if job_id not in job_db:
            return {'status': 'UNKNOWN', 'error': 'Job ID not found'}

        job_info = job_db[job_id]
        remote_path = job_info['remote_path']

        # Some jobs (Align) write to a per-algorithm log file rather
        # than the class-level LOG_FILE. Resolve it once and reuse for
        # both the progress parse and the "Latest log" tail.
        log_file = self._resolve_log_file(job_info)
        log_operations = self._resolve_log_operations(job_info)

        # Check process
        pid = job_info['pid']
        ps_result = self.conn.run(f'ps -p {pid} -o pid=', warn=True, hide=True)
        is_running = ps_result.ok

        # Read status file
        status_result = self.conn.run(
            f'cat {remote_path}/status.txt 2>/dev/null || echo "NO_STATUS"',
            hide=True, warn=True
        )
        status_lines = status_result.stdout.strip().split('\n')

        status = self._infer_status(job_info, is_running, status_lines)

        # Calculate runtime
        submitted_at = datetime.fromisoformat(job_info['submitted_at'])
        runtime = str(datetime.now() - submitted_at).split('.')[0]

        # Parse progress if running
        progress = {}
        if status in ['RUNNING', 'SUBMITTED'] and log_file:
            log_result = self.conn.run(
                f'tail -n 100 {remote_path}/{log_file} 2>/dev/null || echo "No log file"',
                hide=True, warn=True
            )
            progress = self._parse_log_progress(log_result.stdout, log_operations)

        # Build stages from LOG_OPERATIONS
        stages = []
        current_op = progress.get('current_operation')
        if log_operations:
            found_current = False
            for _keyword, label in log_operations:
                if label == current_op:
                    stages.append({'label': label, 'state': 'active'})
                    found_current = True
                elif found_current or current_op is None:
                    if status == 'COMPLETED':
                        stages.append({'label': label, 'state': 'done'})
                    else:
                        stages.append({'label': label, 'state': 'pending'})
                else:
                    stages.append({'label': label, 'state': 'done'})

        # Get last *informative* log line for running jobs.
        # MMseqs2 echoes its full parameter set near the top of each
        # subcommand invocation — lines like "Translation mode 0",
        # "Threads 8", "Min seq id 0.150" — which leak through a naive
        # tail and make the modal look stuck on parameter dumps. We pull
        # a wider window and pick the most recent line that actually
        # describes activity (anything with a colon, a fraction, or a
        # sentence-like form), falling back to the raw tail.
        last_log_line = None
        if status in ['RUNNING', 'SUBMITTED'] and log_file:
            log_tail = self.conn.run(
                f"tail -n 30 {remote_path}/{log_file} 2>/dev/null | "
                r"sed 's/\x1b\[[0-9;]*m//g' | grep -v '^\s*$'",
                hide=True, warn=True
            )
            last_log_line = self._pick_informative_log_line(log_tail.stdout)

        # Update local DB. Same locking discipline as `status()` —
        # lock only the in-memory mutation, not the SSH calls above.
        with self._mutate_job_db() as db:
            entry = db.get(job_id)
            if entry is not None:
                entry['status'] = status
                entry['last_checked'] = datetime.now().isoformat()

        return {
            'job_id': job_id,
            'name': job_info['name'],
            'status': status,
            'runtime': runtime,
            'job_type': job_info.get('job_type', self.JOB_TYPE),
            'database': job_info.get('database', ''),
            'preset': job_info.get('preset', ''),
            'stages': stages,
            'last_log_line': last_log_line,
            **progress
        }

    # ── Hooks for per-job log resolution ────────────────────────────

    def _resolve_log_file(self, job_info: Dict) -> str:
        """Return the log file for this specific job. Defaults to the
        class-level LOG_FILE; subclasses override when the file name
        depends on per-job parameters (e.g., Align picks `clustalo.log`
        / `mafft.log` / `muscle.log` from the chosen algorithm)."""
        return self.LOG_FILE

    def _resolve_log_operations(self, job_info: Dict) -> list:
        """Return the LOG_OPERATIONS list for this specific job. Same
        rationale as `_resolve_log_file` — Align ships a different stage
        list per algorithm."""
        return self.LOG_OPERATIONS

    def _parse_log_progress(self, log_content: str,
                            log_operations: Optional[list] = None) -> Dict:
        """Parse log file for progress information.

        `log_operations` is the keyword/label list to scan for the
        current stage. Defaults to `self.LOG_OPERATIONS` so existing
        callers (no extra arg) keep their behavior; the modal passes
        the per-job list resolved via `_resolve_log_operations`.
        """
        progress = {
            'current_step': None,
            'prefilter_step': None,
            'total_prefilter_steps': None,
            'align_step': None,
            'total_align_steps': None,
            'current_operation': None,
        }

        if not log_content or log_content == "No log file":
            return progress

        lines = log_content.strip().split('\n')

        # Parse prefiltering step.
        prefilter_pattern = r'Process prefiltering step (\d+) of (\d+)'
        for line in reversed(lines):
            match = re.search(prefilter_pattern, line)
            if match:
                progress['prefilter_step'] = int(match.group(1))
                progress['total_prefilter_steps'] = int(match.group(2))
                progress['current_step'] = 'prefilter'
                break

        # Parse alignment progress. MMseqs2 prints both
        # "[X / Y]" tick lines and "X out of Y" summary lines during
        # the `align` subcommand; surface whichever shows up most
        # recently so the modal can render a real percentage.
        align_pattern = r'(\d+)\s*(?:/|of|out of)\s*(\d+)'
        for line in reversed(lines):
            if any(k in line for k in (
                'Calculation of alignments',
                'Compute Smith-Waterman alignments',
                'aligned',
            )):
                continue  # banner-style lines, not counters
            # Only treat numeric ratios as alignment progress when the
            # nearby context says it's alignment — guards against
            # picking up unrelated counters like "step 1 of N" prefilter
            # lines we already parsed above.
            if 'align' not in line.lower():
                continue
            match = re.search(align_pattern, line)
            if match:
                cur = int(match.group(1))
                tot = int(match.group(2))
                if tot > 0 and cur <= tot:
                    progress['align_step'] = cur
                    progress['total_align_steps'] = tot
                    progress['current_step'] = 'align'
                    break

        # Determine current operation using the (possibly per-job)
        # operation list. `lines[-50:]` reversed walks the most recent
        # log lines from newest to oldest, so the matched stage reflects
        # the current activity rather than something from the start of
        # a long-running job.
        operations = log_operations if log_operations is not None else self.LOG_OPERATIONS
        for line in reversed(lines[-50:]):
            for keyword, operation in operations:
                if keyword in line:
                    progress['current_operation'] = operation
                    return progress

        return progress

    # Lines that are pure parameter echoes — MMseqs2 dumps the resolved
    # value of every flag at the top of each subcommand invocation
    # ("Translation mode 0", "Threads 8", "Min seq id 0.150"). Filter
    # them out of the "Latest log" display so the user sees real
    # activity instead of the most recent parameter dump.
    #
    # Constraints, tightened from a looser earlier version that
    # over-matched lines like "Number aligned 12345":
    #   - the title must be 1-3 capitalized words (each word starts
    #     uppercase, then optional lowercase / digits / hyphens). This
    #     matches MMseqs2's title format ("Index table fill", "Min
    #     seq id") but rejects sentence-style lines that begin with
    #     "Number " or "Computing " followed by lowercase content.
    #   - the value must be one short token (no spaces).
    #   - words like "of", "to", "and" anywhere in the title disqualify
    #     it — MMseqs2 only uses those in real activity messages
    #     ("Number of alignments", "Time for processing").
    _PARAM_ECHO_RE = re.compile(
        r'^([A-Z][a-z0-9\-]*)(?:\s+([A-Z][a-z0-9\-]*|[a-z]{1,4}))?'
        r'(?:\s+([A-Z][a-z0-9\-]*|[a-z]{1,4}))?'
        r'\s+(\d+(?:\.\d+)?(?:e[+\-]?\d+)?|true|false|nan|null)\s*$'
    )
    # Connectives that show up only in real activity lines; if any of
    # these appear, the line is informative and we keep it.
    _ACTIVITY_KEYWORDS = (
        " of ", " for ", " to ", " from ", " and ", " in ", " on ",
        " into ", " out ", " with ", "Number ", "Time ",
    )

    def _pick_informative_log_line(self, log_text: str) -> Optional[str]:
        """Return the most recent log line that's not a parameter echo."""
        if not log_text:
            return None
        candidates = [ln for ln in log_text.strip().split('\n') if ln.strip()]
        if not candidates:
            return None
        for line in reversed(candidates):
            stripped = line.strip()
            # Activity-keyword whitelist runs before the param-echo
            # regex so an informative line that happens to look like
            # "Number aligned 12345" is preserved.
            if any(k in stripped for k in self._ACTIVITY_KEYWORDS):
                return stripped
            if self._PARAM_ECHO_RE.match(stripped):
                continue
            # Empty headers like "=================" aren't useful either.
            if set(stripped) <= {'=', '-', ' '}:
                continue
            return stripped
        # Everything was a parameter echo — return None and let the
        # modal render `(idle)` rather than a stale parameter dump that
        # makes a long-finished job look like it's still echoing config.
        return None

    def print_detailed_status(self, job_id: str, watch: bool = False, animation_frame: int = 0):
        """Print formatted status with optional live updates

        .. deprecated::
            Use ``beak status -v`` or ``beak.cli.display.print_status()`` instead.
        """
        import warnings
        warnings.warn(
            "print_detailed_status() is deprecated. Use 'beak status -v' or "
            "beak.cli.display.print_status() instead.",
            DeprecationWarning, stacklevel=2
        )
        spinner = ['\u280b', '\u2819', '\u2839', '\u2838', '\u283c', '\u2834', '\u2826', '\u2827', '\u2807', '\u280f']

        if watch and animation_frame > 0:
            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except ImportError:
                print('\033[2J\033[H')

        status = self.detailed_status(job_id)

        print(f"\n{'='*60}")
        print(f"Job: {status['name']} ({status['job_id']})")
        print(f"Type: {self.JOB_TYPE.title()} | Status: {status['status']} | Runtime: {status['runtime']}")
        print(f"{'='*60}")

        if status['status'] in ['RUNNING', 'SUBMITTED']:
            if status.get('prefilter_step') and status.get('total_prefilter_steps'):
                total = status['total_prefilter_steps']
                current = status['prefilter_step']
                pct = (current / total) * 100

                bar_length = 40
                filled = int(bar_length * current / total)
                bar = '\u2588' * filled + '\u2591' * (bar_length - filled)

                icon = spinner[animation_frame % len(spinner)] if watch else '\u27f3'
                print(f"\n{icon} Prefilter: Step {current}/{total} ({pct:.1f}%)")
                print(f"  [{bar}]")

            if status.get('current_operation'):
                print(f"\n\U0001f504 {status['current_operation']}")

            if watch:
                print(f"\n(Press Ctrl+C to stop watching)")

        print(f"{'='*60}\n")

        if watch and status['status'] in ['RUNNING', 'SUBMITTED']:
            try:
                time.sleep(1)
                self.print_detailed_status(job_id, watch=True, animation_frame=animation_frame + 1)
            except KeyboardInterrupt:
                print("\nStopped watching.")

    # ── Wait ────────────────────────────────────────────────────────

    def wait(self, job_id: str, check_interval: int = 30,
             verbose: bool = True, show_progress: bool = False):
        """
        Wait for job completion

        Args:
            job_id: Job identifier
            check_interval: Seconds between checks
            verbose: Print updates
            show_progress: Show detailed progress (uses print_detailed_status)
        """
        if show_progress:
            self.print_detailed_status(job_id, watch=True)
            return self.status(job_id)['status']

        if verbose:
            print(f"Waiting for job {job_id}...")

        while True:
            status_info = self.status(job_id)

            if verbose:
                print(f"  [{status_info['runtime']}] {status_info['status']}")

            if status_info['status'] in ['COMPLETED', 'FAILED', 'UNKNOWN', 'CANCELLED']:
                break

            time.sleep(check_interval)

        if verbose:
            if status_info['status'] == 'COMPLETED':
                print(f"✓ Job completed in {status_info['runtime']}")
            else:
                print(f"✗ Job {status_info['status'].lower()}")

        return status_info['status']

    # ── List jobs ───────────────────────────────────────────────────

    def list_jobs(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """
        List jobs of this type, optionally filtered by status

        Args:
            status_filter: Optional status to filter by (RUNNING, COMPLETED, etc.)

        Returns:
            DataFrame with job information
        """
        job_db = self._load_job_db()

        # Filter by job type
        typed_jobs = {k: v for k, v in job_db.items()
                      if v.get('job_type') == self.JOB_TYPE}

        if not typed_jobs:
            return pd.DataFrame()

        jobs_list = []
        for job_id, info in typed_jobs.items():
            current_status = self.status(job_id)

            if status_filter and current_status['status'] != status_filter:
                continue

            row = {
                'job_id': job_id,
                'name': info['name'],
                'status': current_status['status'],
                'submitted': info['submitted_at'],
                'runtime': current_status['runtime']
            }
            row.update(self._list_jobs_extra_columns(info))
            jobs_list.append(row)

        df = pd.DataFrame(jobs_list)
        if not df.empty:
            df = df.sort_values('submitted', ascending=False)

        return df

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        """Return extra columns for list_jobs. Override in subclasses."""
        return {}

    # ── Cancel / Cleanup / Log ──────────────────────────────────────

    def cancel(self, job_id: str):
        """Cancel a running job and its entire descendant tree.

        The launch script is `nohup bash run.sh &` and the actual worker
        (mmseqs / clustalo / mafft / muscle / docker exec) runs as a
        forked child. Killing only the recorded wrapper PID leaves the
        worker orphaned and still consuming CPU and memory — which is
        exactly what makes a runaway alignment "gum up the server." We
        walk descendants via `pgrep -P`, SIGTERM the whole tree, give it
        2s to flush, then SIGKILL anything still alive.
        """
        job_db = self._load_job_db()

        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")

        pid = job_db[job_id]['pid']

        kill_tree = (
            'descendants() { '
            '  local p=$1 kids; '
            '  kids=$(pgrep -P "$p" 2>/dev/null); '
            '  for k in $kids; do descendants "$k"; echo "$k"; done; '
            '}; '
            f'ALL=$(descendants {pid}; echo {pid}); '
            'kill -TERM $ALL 2>/dev/null; sleep 2; '
            'kill -KILL $ALL 2>/dev/null; true'
        )
        self.conn.run(kill_tree, warn=True, hide=True)

        remote_path = job_db[job_id]['remote_path']
        self.conn.run(
            f'echo "Job cancelled: $(date)" >> {remote_path}/status.txt && '
            f'echo "CANCELLED" >> {remote_path}/status.txt',
            hide=True, warn=True,
        )

        # Reflect the cancellation locally so the layers panel and
        # `beak jobs` don't continue showing RUNNING until the next poll
        # re-derives status from the remote.
        with self._mutate_job_db() as db:
            entry = db.get(job_id)
            if entry is not None:
                entry['status'] = 'CANCELLED'
                entry['last_checked'] = datetime.now().isoformat()

        print(f"✓ Job {job_id} cancelled")

    def cleanup(self, job_id: str, keep_results: bool = False):
        """Clean up remote job files.

        Safety invariants — destroying the wrong directory on a
        workstation is unrecoverable, so the strict checks below run
        client-side AND the rm itself is gated server-side:

        1. The recorded `remote_path` must equal exactly
           `f"{self.remote_job_dir}/{job_id}"`. beak's submit codepaths
           always produce that form; any deviation means the manifest
           is wrong or has been edited, and we refuse rather than nuke.
        2. The path is `shlex.quote`d and the rm uses `--` so a
           tampered path with spaces, leading dashes, or shell
           metacharacters can't reroute the command.
        3. Server-side we require `[ -d "$p" ]` (directory exists) AND
           `[ ! -L "$p" ]` (not a symlink). The symlink check stops
           `rm -rf` from following a manual symlink to anywhere else
           on the host.

        These are belt-and-braces — the `remote_path == scratch_root /
        job_id` invariant alone should be sufficient on its own, but
        the cost of the extra guards is one shell test per cleanup.
        """
        import shlex

        job_db = self._load_job_db()

        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")

        remote_path = (job_db[job_id] or {}).get('remote_path') or ''
        if not remote_path:
            raise ValueError(
                f"Job {job_id} has no remote_path recorded — refusing "
                f"to cleanup."
            )

        # Strict-equality check against the expected scratch path so a
        # corrupt or hand-edited manifest can't redirect the rm.
        expected = f"{self.remote_job_dir.rstrip('/')}/{job_id}"
        if remote_path.rstrip('/') != expected:
            raise ValueError(
                f"Refusing to cleanup {remote_path!r}: does not match "
                f"the expected scratch path {expected!r}. (If this is "
                f"intentional, run the rm by hand on the remote.)"
            )

        quoted = shlex.quote(remote_path)

        if keep_results:
            # Targeted cleanup of just the per-job `tmp/` subdir.
            self.conn.run(
                'set -e; '
                f'p={quoted}/tmp; '
                'if [ -d "$p" ] && [ ! -L "$p" ]; then rm -rf -- "$p"; fi',
                hide=True, warn=True,
            )
            print(f"✓ Cleaned up temp files for job {job_id}")
        else:
            self.conn.run(
                'set -e; '
                f'p={quoted}; '
                'if [ -d "$p" ] && [ ! -L "$p" ]; then rm -rf -- "$p"; fi',
                hide=True, warn=True,
            )
            with self._mutate_job_db() as db:
                db.pop(job_id, None)
            print(f"✓ Cleaned up job {job_id}")

    def get_log(self, job_id: str, lines: int = 50):
        """Get the last N lines of job log for debugging"""
        job_db = self._load_job_db()
        if job_id not in job_db:
            print(f"Job {job_id} not found")
            return

        remote_path = job_db[job_id]['remote_path']

        files = self.conn.run(f'ls -la {remote_path}/', hide=True, warn=True)
        print("Files in job directory:")
        print(files.stdout)

        log = self.conn.run(
            f'tail -n {lines} {remote_path}/{self.LOG_FILE} 2>&1 || echo "No log file"',
            hide=True, warn=True
        )
        print(f"\n{self.LOG_FILE} (last {lines} lines):")
        print(log.stdout)

        nohup = self.conn.run(
            f'cat {remote_path}/nohup.out 2>&1 || echo "No nohup file"',
            hide=True, warn=True
        )
        print(f"\nNohup output:")
        print(nohup.stdout)

    def follow_log(self, job_id: str, interval: float = 2.0):
        """Stream the job log live, like `tail -f` — Ctrl-C to stop.

        Polls the remote log every `interval` seconds using a byte offset,
        so new lines appear without reprinting what was already shown.
        Exits cleanly when the job reaches a terminal state.
        """
        import time

        job_db = self._load_job_db()
        if job_id not in job_db:
            print(f"Job {job_id} not found")
            return

        remote_path = job_db[job_id]['remote_path']
        log_file = f"{remote_path}/{self.LOG_FILE}"

        # Show the last 20 lines first so the user has context, then stream.
        initial = self.conn.run(
            f'tail -n 20 {log_file} 2>&1 || echo "(no log yet)"',
            hide=True, warn=True,
        )
        print(initial.stdout, end='')

        # Offset starts at the current size so subsequent polls only emit new bytes.
        size_result = self.conn.run(
            f'wc -c < {log_file} 2>/dev/null || echo 0',
            hide=True, warn=True,
        )
        try:
            offset = int(size_result.stdout.strip() or 0)
        except ValueError:
            offset = 0

        try:
            while True:
                # Read any new bytes appended since the last offset.
                chunk = self.conn.run(
                    f'tail -c +{offset + 1} {log_file} 2>/dev/null',
                    hide=True, warn=True,
                )
                if chunk.stdout:
                    print(chunk.stdout, end='', flush=True)
                    offset += len(chunk.stdout.encode('utf-8', errors='replace'))

                # Stop once the job has finished.
                info = self.status(job_id)
                if info.get('status') in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n(stopped)")

    # ── Docker support ──────────────────────────────────────────────

    def _resolve_docker_dir(self) -> tuple:
        """Resolve the Docker service directory and compose project name.

        Reads [docker] service_dir from config if set (shared across all
        beak users on the same remote); otherwise falls back to a
        per-user path under remote_job_dir.

        Returns:
            (docker_dir, project_name, is_shared)
        """
        from ..config import get_docker_config

        docker_cfg = get_docker_config()
        shared_dir = docker_cfg['service_dir']
        project_name = docker_cfg['project_name']

        if shared_dir:
            return shared_dir, project_name, True
        return f"{self.remote_job_dir}/docker", project_name, False

    def _ensure_docker_service(self, service_name: str = "embeddings"):
        """Ensure the Docker service is deployed, up to date, and running.

        Always re-uploads the local Docker source files (they're tiny) and
        runs `up -d --build` so source changes land on every submission.
        Docker layer caching makes this near-free when nothing changed, and
        it prevents the "remote dir exists but some file is missing / stale"
        failure mode where the build fails but the service never redeploys.

        When [docker] service_dir is configured, the service is shared across
        all beak users on the remote — the first user triggers the build,
        everyone else's `docker compose ps` sees it already running and
        short-circuits to exec-only.
        """
        docker_dir, project_name, is_shared = self._resolve_docker_dir()
        compose = f"docker compose --project-name {project_name}"

        # Preflight: fail helpfully if a shared dir is configured but unwritable
        if is_shared:
            writable = self.conn.run(
                f'mkdir -p {docker_dir} 2>/dev/null && [ -w {docker_dir} ] && echo OK || echo NO',
                hide=True, warn=True,
            )
            if writable.stdout.strip() != 'OK':
                raise PermissionError(
                    f"Shared Docker service dir {docker_dir!r} is not writable "
                    f"by user {self.conn.user!r}. Ask the admin to grant group "
                    f"write access, or unset `docker.service_dir` in your config "
                    f"to use a per-user deployment at {self.remote_job_dir}/docker."
                )
        else:
            self.conn.run(f'mkdir -p {docker_dir}', hide=True)

        self._upload_docker_files(docker_dir)

        # Always run `up -d --build`. Docker layer caching makes this near-
        # free when nothing changed (a few seconds of compose overhead), and
        # it's the only way changes to the uploaded source files actually
        # land in the running container — COPY instructions only re-execute
        # when the build runs. A previous short-circuit skipped the rebuild
        # when the service was already running, which meant source-level
        # bug fixes silently failed to deploy.
        where = "shared" if is_shared else "per-user"
        print(f"Starting Docker service '{service_name}' ({where}, build if needed)...")
        with self.conn.cd(docker_dir):
            result = self.conn.run(
                f'{compose} up -d --build',
                hide=True, warn=True,
            )
        if not result.ok:
            print("✗ Docker service failed to start. Build output:\n")
            print(result.stdout)
            if result.stderr.strip():
                print("\nstderr:\n" + result.stderr)
            raise RuntimeError(
                f"Docker service '{service_name}' failed to start "
                f"(exit {result.exited}). See output above."
            )

        # Wait for the container to actually be ready for exec.
        # `up -d` returns as soon as the container is *started*, but the
        # runtime may still be booting its PID 1 — any immediate `exec` can
        # hit "OCI runtime exec failed: procReady not received".
        self._wait_for_container_ready(docker_dir, project_name, service_name)
        print(f"✓ Docker service '{service_name}' is up")

    def _wait_for_container_ready(self, docker_dir: str, project_name: str,
                                  service_name: str, max_wait_s: int = 60):
        """Poll `docker compose exec <service> true` until it succeeds.

        Each probe has its own 5s `timeout` so a hung Docker API doesn't
        wedge the whole loop — we always make forward progress and either
        succeed or surface a descriptive error pointing at `docker ps -a`
        on the remote.
        """
        # Also include `docker compose ps` output on failure so the user
        # can tell whether the container is Up / Restarting / Exited.
        with self.conn.cd(docker_dir):
            result = self.conn.run(
                f'for i in $(seq 1 {max_wait_s // 2}); do '
                f'  if timeout 5s docker compose --project-name {project_name} '
                f'exec -T {service_name} true 2>/dev/null; then '
                f'    echo READY; exit 0; '
                f'  fi; '
                f'  sleep 2; '
                f'done; '
                f'echo TIMEOUT; '
                f'docker compose --project-name {project_name} ps 2>&1 || true',
                hide=True, warn=True,
            )
        if 'READY' not in (result.stdout or ''):
            raise RuntimeError(
                f"Docker service '{service_name}' started but the container "
                f"was not ready for exec within {max_wait_s}s.\n"
                f"{result.stdout.strip()}\n"
                f"Try a clean recreate:\n"
                f"  ssh <remote> 'cd {docker_dir} && "
                f"docker compose --project-name {project_name} "
                f"down --remove-orphans && "
                f"docker compose --project-name {project_name} up -d --build'"
            )

    def _upload_docker_files(self, docker_dir: str):
        """Upload the Docker source bundle from the installed package to the remote.

        Missing optional files (e.g., .dockerignore) are silently skipped;
        the Dockerfile and generate_embeddings.py are required.
        """
        self.conn.run(f'mkdir -p {docker_dir}', hide=True)

        # Prefer importlib.resources (stdlib) over the deprecated pkg_resources.
        try:
            from importlib.resources import files as _resource_files
            docker_files_dir = Path(str(_resource_files('beak').joinpath('remote/docker')))
        except (ImportError, AttributeError):
            # Python < 3.9 fallback
            from pkg_resources import resource_filename
            docker_files_dir = Path(resource_filename('beak', 'remote/docker'))

        required = ['Dockerfile', 'docker-compose.yml', 'generate_embeddings.py']
        optional = ['requirements.txt', '.dockerignore']

        for filename in required:
            local_path = docker_files_dir / filename
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Required Docker source file missing from package: {local_path}"
                )
            self.conn.put(str(local_path), f"{docker_dir}/{filename}")

        for filename in optional:
            local_path = docker_files_dir / filename
            if local_path.exists():
                self.conn.put(str(local_path), f"{docker_dir}/{filename}")

    def _deploy_docker_service(self, service_name: str = "embeddings"):
        """Alias retained for compatibility; delegates to _ensure_docker_service."""
        self._ensure_docker_service(service_name)

    def _docker_exec(self, command: str, service_name: str = "embeddings"):
        """Execute a command inside the Docker container for this service."""
        docker_dir, project_name, _ = self._resolve_docker_dir()
        with self.conn.cd(docker_dir):
            return self.conn.run(
                f'docker compose --project-name {project_name} exec -T {service_name} {command}',
                hide=True,
            )
