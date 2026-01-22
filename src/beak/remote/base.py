
import json
import shutil
import pandas as pd
from pathlib import Path

from fabric import Connection
from datetime import datetime
from typing import Dict, Optional

class ProjectManager:
    """Manage local beak project organization"""
    
    def __init__(self, projects_dir: Optional[Path] = None):
        """
        Initialize ProjectManager
        
        Args:
            projects_dir: Path to projects directory (defaults to ~/beak_projects)
        """
        self.projects_dir = projects_dir or Path.home() / "beak_projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.projects_dir / ".index.json"
    
    def _get_index(self) -> Dict:
        """Load projects index"""
        if not self.index_file.exists():
            return {}
        with open(self.index_file) as f:
            return json.load(f)
    
    def _save_index(self, index: Dict):
        """Save projects index"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def organize_project(self, 
                    job_ids: list[str],
                    project_name: str,
                    description: Optional[str] = None,
                    cleanup: bool = False) -> Path:
        """
        Organize multiple related jobs into a single coherent project
        
        Args:
            job_ids: List of job IDs to organize together
            project_name: Name for the organized project
            description: Optional project description
            cleanup: If True, remove original job directories after organizing
        
        Returns:
            Path to the organized project directory
        """
        from shutil import copytree, rmtree
        
        # Create organized project directory
        organized_dir = self.projects_dir / project_name
        if organized_dir.exists():
            raise ValueError(f"Project '{project_name}' already exists at {organized_dir}")
        
        organized_dir.mkdir(parents=True)
        
        # Load index
        index = self._get_index()
        
        # Organize each job into subdirectories
        job_metadata = []
        dirs_to_cleanup = []  # Track directories to remove
        
        for job_id in job_ids:
            # Find the job's project directory
            if job_id not in index:
                print(f"Warning: Job {job_id} not found in index, skipping")
                continue
            
            job_info = index[job_id]
            source_dir = Path(job_info['project_dir'])
            
            # If source doesn't exist, it might be in an old organized project
            # Try to find the standalone directory
            if not source_dir.exists():
                job_type = job_info.get('job_type', 'unknown')
                potential_dir = self.projects_dir / f"{job_type}_{job_id}"
                
                if potential_dir.exists():
                    print(f"Using standalone directory: {potential_dir.name}")
                    source_dir = potential_dir
                else:
                    # Check if it's in an old organized project that still exists
                    old_organized = job_info.get('organized_project')
                    if old_organized:
                        old_organized_path = Path(job_info['project_dir'])
                        if old_organized_path.exists():
                            print(f"Found in organized project: {old_organized}/{old_organized_path.name}")
                            source_dir = old_organized_path
                        else:
                            print(f"Warning: Job {job_id} is organized in '{old_organized}' which doesn't exist")
                            print(f"  Looking for standalone directory...")
                            if potential_dir.exists():
                                source_dir = potential_dir
                            else:
                                print(f"  Not found, skipping")
                                continue
                    else:
                        print(f"Warning: Project directory {source_dir} not found, skipping")
                        continue
            
            if not source_dir.exists():
                print(f"Warning: Could not locate files for job {job_id}, skipping")
                continue
            
            job_type = job_info.get('job_type', 'unknown')
            
            # Create subdirectory named by job type
            # If multiple of same type, add index
            target_subdir = organized_dir / job_type
            counter = 1
            while target_subdir.exists():
                target_subdir = organized_dir / f"{job_type}_{counter}"
                counter += 1
            
            # Copy the directory
            print(f"Copying {source_dir.name} -> {project_name}/{target_subdir.name}")
            copytree(source_dir, target_subdir)
            
            # Track for cleanup if requested
            if cleanup:
                dirs_to_cleanup.append(source_dir)
            
            # Collect metadata
            job_metadata.append({
                'job_id': job_id,
                'job_type': job_type,
                'original_name': job_info.get('name'),
                'subdirectory': target_subdir.name,
                'created': job_info.get('created'),
            })
            
            # Update index to point to new location
            index[job_id]['project_dir'] = str(target_subdir)
            index[job_id]['organized_project'] = project_name

        # Create project README
        readme_content = f"""# {project_name}

    {description or 'No description provided'}

    ## Jobs Included

    """
        for meta in job_metadata:
            readme_content += f"""
    ### {meta['job_type'].upper()}: {meta['original_name']}
    - **Job ID**: `{meta['job_id']}`
    - **Directory**: `{meta['subdirectory']}/`
    - **Created**: {meta.get('created', 'Unknown')}
    """
        
        readme_file = organized_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create project metadata
        project_metadata = {
            'project_name': project_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'jobs': job_metadata
        }
        
        metadata_file = organized_dir / "project_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(project_metadata, f, indent=2)
        
        # Save updated index
        self._save_index(index)
        
        # Cleanup original directories if requested
        if cleanup and dirs_to_cleanup:
            print(f"\nCleaning up {len(dirs_to_cleanup)} original job directories...")
            for dir_path in dirs_to_cleanup:
                print(f"  Removing {dir_path.name}")
                rmtree(dir_path)
            print("✓ Cleanup complete")
        
        print(f"\n✓ Organized project created: {project_name}")
        print(f"  Location: {organized_dir}")
        print(f"  Jobs: {len(job_metadata)}")
        print(f"\nProject structure:")
        
        # Print tree
        for item in sorted(organized_dir.iterdir()):
            if item.is_dir():
                print(f"  {item.name}/")
                for subitem in sorted(item.iterdir()):
                    print(f"    └── {subitem.name}")
            else:
                print(f"  {item.name}")
        
        return organized_dir
    
    def list_projects(self) -> pd.DataFrame:
        """
        List all projects (both individual jobs and organized projects)
        
        Returns:
            DataFrame with project information
        """
        index = self._get_index()
        
        if not index:
            return pd.DataFrame()
        
        projects = []
        for job_id, info in index.items():
            projects.append({
                'job_id': job_id,
                'name': info.get('name'),
                'job_type': info.get('job_type'),
                'created': info.get('created'),
                'organized_project': info.get('organized_project', None),
                'path': info.get('project_dir')
            })
        
        df = pd.DataFrame(projects)
        if not df.empty:
            df = df.sort_values('created', ascending=False)
        
        return df
    
    def list_organized_projects(self) -> pd.DataFrame:
        """
        List all organized projects
        
        Returns:
            DataFrame with organized project information
        """
        projects = []
        
        for item in self.projects_dir.iterdir():
            if not item.is_dir():
                continue
            
            metadata_file = item / "project_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                projects.append({
                    'project_name': metadata['project_name'],
                    'description': metadata.get('description', ''),
                    'created': metadata['created_at'],
                    'num_jobs': len(metadata['jobs']),
                    'job_types': ', '.join(set(j['job_type'] for j in metadata['jobs'])),
                    'path': str(item)
                })
        
        if not projects:
            return pd.DataFrame()
        
        return pd.DataFrame(projects).sort_values('created', ascending=False)
    
    def get_project_info(self, project_name: str) -> Dict:
        """
        Get detailed information about an organized project
        
        Args:
            project_name: Name of the organized project
        
        Returns:
            Dictionary with project metadata
        """
        project_dir = self.projects_dir / project_name
        metadata_file = project_dir / "project_metadata.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Project '{project_name}' not found or not an organized project")
        
        with open(metadata_file) as f:
            return json.load(f)
        

class RemoteJobManager:
    """Base class for remote job management"""
    
    LOCAL_PROJECTS_DIR = Path.home() / "beak_projects"

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
        self.local_job_db = Path.home() / ".beak" / "jobs.json"
        
        # Setup
        self._setup_remote_dir()
        self._setup_local_db()
        self._setup_projects_dir()
    
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

    def _setup_projects_dir(self):
        """Setup local projects directory with index"""
        self.LOCAL_PROJECTS_DIR.mkdir(exist_ok=True)
        index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
        if not index_file.exists():
            index_file.write_text(json.dumps({}, indent=2))

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
        """
        Create local project directory
        
        Args:
            job_id: Unique job identifier
            job_type: 'search', 'taxonomy', 'pipeline', etc.
            name: Optional human-readable name
            query_file: Optional query file to copy to project
        
        Returns:
            Path to created project directory
        """
        # Generate project name
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
        
        # Create metadata
        metadata = {
            'job_id': job_id,
            'job_type': job_type,
            'name': name or project_name,
            'created': datetime.now().isoformat()
        }
        
        with open(project_dir / ".metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy input file if provided
        if query_file:
            shutil.copy2(query_file, project_dir / "input.fasta")
        
        # Update index
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
        
        # Remove directory
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        # Remove from index
        index = self._get_projects_index()
        if job_id in index:
            del index[job_id]
            index_file = self.LOCAL_PROJECTS_DIR / ".index.json"
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
        
        # Remove from jobs DB
        job_db = self._load_job_db()
        if job_id in job_db:
            del job_db[job_id]
            self._save_job_db(job_db)
        
        print(f"✓ Deleted project {job_id}")
    
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

    def _ensure_docker_service(self, service_name: str = "embeddings"):
        """Ensure Docker service is deployed and running on remote"""
        docker_dir = f"{self.remote_job_dir}/docker"
        
        # Check if docker directory exists
        dir_check = self.conn.run(
            f'[ -d {docker_dir} ] && echo "EXISTS" || echo "NOT_FOUND"',
            hide=True, warn=True
        )
        
        if dir_check.stdout.strip() == "NOT_FOUND":
            print(f"Docker service not deployed. Deploying now...")
            self._deploy_docker_service(service_name)
            return
        
        # Check if service is running
        with self.conn.cd(docker_dir):
            check = self.conn.run(
                f'docker compose ps --services --filter status=running | grep {service_name}',
                hide=True, warn=True
            )
        
        if not check.ok or service_name not in check.stdout:
            print(f"Starting Docker service '{service_name}'...")
            with self.conn.cd(docker_dir):
                self.conn.run('docker compose up -d', hide=True)
            print(f"✓ Docker service started")
    
    def _deploy_docker_service(self, service_name: str = "embeddings"):
        """Deploy Docker service to remote"""
        from pkg_resources import resource_filename
        
        docker_dir = f"{self.remote_job_dir}/docker"
        self.conn.run(f'mkdir -p {docker_dir}', hide=True)
        
        # Upload Docker files
        docker_files_dir = resource_filename('beak', 'remote/docker')
        
        # List of all files to upload
        files_to_upload = [
            'Dockerfile',
            'docker-compose.yml',
            '.dockerignore',
            'requirements.txt',
            'generate_embeddings.py'  # ← ADD THIS
        ]
        
        for filename in files_to_upload:
            local_path = Path(docker_files_dir) / filename
            if local_path.exists():
                self.conn.put(str(local_path), f"{docker_dir}/{filename}")
                print(f"  Uploaded {filename}")
            else:
                print(f"  WARNING: {filename} not found at {local_path}")
        
        # Build and start
        with self.conn.cd(docker_dir):
            print("  Building Docker image...")
            self.conn.run('docker compose build', hide=False)
            
            print("  Starting service...")
            self.conn.run('docker compose up -d', hide=False)
        
        print(f"✓ Docker service '{service_name}' deployed")
    
    def _docker_exec(self, command: str, service_name: str = "embeddings"):
        """Execute command inside Docker container"""
        docker_dir = f"{self.remote_job_dir}/docker"
        with self.conn.cd(docker_dir):
            return self.conn.run(
                f'docker compose exec -T {service_name} {command}',
                hide=True
            )