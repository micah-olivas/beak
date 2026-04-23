"""Base class for remote job management with shared lifecycle methods"""

import re
import json
import time
import shutil
import pandas as pd
from pathlib import Path

from fabric import Connection
from datetime import datetime
from typing import Dict, Optional


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

            self.conn = Connection(
                host=host,
                user=user,
                connect_timeout=10,
                connect_kwargs={"key_filename": str(Path(key_path).expanduser())}
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
        """Load local job database"""
        with open(self.local_job_db, 'r') as f:
            return json.load(f)

    def _save_job_db(self, db: Dict):
        """Save local job database"""
        with open(self.local_job_db, 'w') as f:
            json.dump(db, f, indent=2)

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

        Handles the ambiguous "no status.txt yet" case: within a short grace
        window right after submission it's a genuine SUBMITTED (script
        staging); past that, if the PID is dead too, the submission script
        died before writing any status and we surface it as FAILED rather
        than leave it stuck as SUBMITTED forever.
        """
        if 'COMPLETED' in status_lines:
            return 'COMPLETED'
        if 'FAILED' in status_lines:
            return 'FAILED'
        if 'RUNNING' in status_lines or is_running:
            return 'RUNNING'

        if status_lines and status_lines[0] == 'NO_STATUS':
            submitted_at_str = job_info.get('submitted_at')
            if submitted_at_str:
                try:
                    submitted_at = datetime.fromisoformat(submitted_at_str)
                    age = (datetime.now() - submitted_at).total_seconds()
                    if age > self._LAUNCH_GRACE_SECONDS:
                        # PID dead + no status file + past grace window ⇒
                        # the launch script itself crashed.
                        return 'FAILED'
                except ValueError:
                    pass
            return 'SUBMITTED'

        return 'UNKNOWN'

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

        # Update local DB
        job_info['status'] = status
        job_info['last_checked'] = datetime.now().isoformat()
        self._save_job_db(job_db)

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
        if status in ['RUNNING', 'SUBMITTED'] and self.LOG_FILE:
            log_result = self.conn.run(
                f'tail -n 100 {remote_path}/{self.LOG_FILE} 2>/dev/null || echo "No log file"',
                hide=True, warn=True
            )
            progress = self._parse_log_progress(log_result.stdout)

        # Build stages from LOG_OPERATIONS
        stages = []
        current_op = progress.get('current_operation')
        if self.LOG_OPERATIONS:
            found_current = False
            for _keyword, label in self.LOG_OPERATIONS:
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

        # Get last log line for running jobs
        last_log_line = None
        if status in ['RUNNING', 'SUBMITTED'] and self.LOG_FILE:
            log_tail = self.conn.run(
                f"tail -n 5 {remote_path}/{self.LOG_FILE} 2>/dev/null | "
                r"sed 's/\x1b\[[0-9;]*m//g' | grep -v '^\s*$' | tail -n 1",
                hide=True, warn=True
            )
            last_log_line = log_tail.stdout.strip() or None

        # Update local DB
        job_info['status'] = status
        job_info['last_checked'] = datetime.now().isoformat()
        self._save_job_db(job_db)

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

    def _parse_log_progress(self, log_content: str) -> Dict:
        """Parse log file for progress information"""
        progress = {
            'current_step': None,
            'prefilter_step': None,
            'total_prefilter_steps': None,
            'current_operation': None,
        }

        if not log_content or log_content == "No log file":
            return progress

        lines = log_content.strip().split('\n')

        # Parse prefiltering step
        prefilter_pattern = r'Process prefiltering step (\d+) of (\d+)'
        for line in reversed(lines):
            match = re.search(prefilter_pattern, line)
            if match:
                progress['prefilter_step'] = int(match.group(1))
                progress['total_prefilter_steps'] = int(match.group(2))
                progress['current_step'] = 'prefilter'
                break

        # Determine current operation using subclass-defined operations
        for line in reversed(lines[:50]):
            for keyword, operation in self.LOG_OPERATIONS:
                if keyword in line:
                    progress['current_operation'] = operation
                    return progress

        return progress

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
        """Cancel a running job"""
        job_db = self._load_job_db()

        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")

        pid = job_db[job_id]['pid']
        self.conn.run(f'kill {pid} 2>/dev/null || true', warn=True, hide=True)

        remote_path = job_db[job_id]['remote_path']
        self.conn.run(
            f'echo "Job cancelled: $(date)" >> {remote_path}/status.txt && '
            f'echo "CANCELLED" >> {remote_path}/status.txt',
            hide=True
        )

        print(f"✓ Job {job_id} cancelled")

    def cleanup(self, job_id: str, keep_results: bool = False):
        """Clean up remote job files"""
        job_db = self._load_job_db()

        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")

        remote_path = job_db[job_id]['remote_path']

        if keep_results:
            self.conn.run(f'rm -rf {remote_path}/tmp', hide=True, warn=True)
            print(f"✓ Cleaned up temp files for job {job_id}")
        else:
            self.conn.run(f'rm -rf {remote_path}', hide=True, warn=True)
            del job_db[job_id]
            self._save_job_db(job_db)
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

        with self.conn.cd(docker_dir):
            check = self.conn.run(
                f'{compose} ps --services --filter status=running | grep {service_name}',
                hide=True, warn=True,
            )

        if check.ok and service_name in check.stdout:
            return  # already running with current source (shared or own)

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
