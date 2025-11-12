from fabric import Connection
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional

class RemoteJobManager:
    """Base class for remote job management"""
    
    DB_BASE_PATH = "/srv/protein_sequence_databases"
    
    def __init__(self, host: str, user: str, key_path: Optional[str] = None, remote_job_dir: Optional[str] = None):
        # Auto-detect SSH key if not provided
        if key_path is None:
            key_path = self._find_ssh_key()
        
        self.conn = Connection(
            host=host,
            user=user,
            connect_kwargs={"key_filename": str(Path(key_path).expanduser())}
        )
        
        # Default remote job directory
        if remote_job_dir is None:
            remote_job_dir = "~/beak_jobs"
        
        # Expand tilde in remote path for SFTP compatibility
        if remote_job_dir.startswith('~'):
            result = self.conn.run('echo $HOME', hide=True)
            remote_job_dir = result.stdout.strip() + remote_job_dir[1:]
        
        self.remote_job_dir = remote_job_dir
        self.local_job_db = Path.home() / ".mmseqs_jobs" / "jobs.json"
        
        # Setup
        self._setup_remote_dir()
        self._setup_local_db()
    
    def _find_ssh_key(self) -> str:
        """Auto-detect SSH key from common locations"""
        ssh_dir = Path.home() / ".ssh"
        
        # Priority order of key types
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
    
    def _setup_remote_dir(self):
        """Ensure remote job directory exists"""
        self.conn.run(f'mkdir -p {self.remote_job_dir}', hide=True)
    
    def _setup_local_db(self):
        """Setup local job tracking database"""
        self.local_job_db.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_job_db.exists():
            self._save_job_db({})
    
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
    
    def status(self, job_id: str) -> Dict:
        """Check status of a job"""
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
            hide=True,
            warn=True
        )
        status_lines = status_result.stdout.strip().split('\n')
        
        if 'COMPLETED' in status_lines:
            status = 'COMPLETED'
        elif 'FAILED' in status_lines:
            status = 'FAILED'
        elif 'RUNNING' in status_lines or is_running:
            status = 'RUNNING'
        elif status_lines[0] == 'NO_STATUS':
            status = 'SUBMITTED'
        else:
            status = 'UNKNOWN'
        
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
            'job_type': job_info.get('job_type', 'unknown')
        }
    
    def cancel(self, job_id: str):
        """Cancel a running job"""
        job_db = self._load_job_db()
        
        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")
        
        pid = job_db[job_id]['pid']
        
        # Kill the process
        self.conn.run(f'kill {pid} 2>/dev/null || true', warn=True, hide=True)
        
        # Update status
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
        
        # Check what files exist
        files = self.conn.run(f'ls -la {remote_path}/', hide=True, warn=True)
        print("Files in job directory:")
        print(files.stdout)
        
        # Get the log
        log = self.conn.run(f'tail -n {lines} {remote_path}/mmseqs.log 2>&1 || echo "No log file"', hide=True, warn=True)
        print(f"\nMMseqs log (last {lines} lines):")
        print(log.stdout)
        
        # Get nohup output
        nohup = self.conn.run(f'cat {remote_path}/nohup.out 2>&1 || echo "No nohup file"', hide=True, warn=True)
        print(f"\nNohup output:")
        print(nohup.stdout)