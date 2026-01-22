import re
import uuid
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union

from .base import RemoteJobManager

class MMseqsSearch(RemoteJobManager):  # Changed from MMseqsBase
    """MMseqs2 search job manager for remote execution"""
    
    # Available databases for search
    AVAILABLE_DBS = {
        'bfd': 'bfd-first_non_consensus_sequences.fasta',
        'gtdb': 'GTDB',
        'swissprot': 'swissprot',
        'trembl': 'TrEMBL',
        'uniprotkb': 'UniProtKB',
        'uniref100': 'UniRef100',
        'uniref90': 'UniRef90',
        'uniref50': 'uniref50',
        'nt_rna': 'nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
        'pdb_seqres': 'pdb_seqres_2022_09_28.fasta',
        'rfam': 'rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
        'rnacentral': 'rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    }

    # Search presets
    PRESETS = {
        'default': {
            'description': 'Default MMseqs2 sensitivity (balanced speed/sensitivity)',
            'params': {
                's': 5.7,  # sensitivity
                'e': 0.001,  # e-value
            }
        },
        'fast': {
            'description': 'Fast search for close homologs',
            'params': {
                's': 4.0,
                'e': 0.001,
            }
        },
        'sensitive': {
            'description': 'More sensitive search (slower)',
            'params': {
                's': 7.5,
                'e': 0.001,
            }
        },
        'exhaustive': {
            'description': 'Exhaustive search for remote homologs',
            'params': {
                's': 8.5,  # very high sensitivity
                'e': 10,  # very permissive e-value
                'max_seqs': 30000,  # allow many results
                'min_seq_id': 0.15,  # very low identity threshold
                'c': 0.3,  # low coverage threshold
                'cov_mode': 0,  # bidirectional coverage
            }
        },
        'very_sensitive': {
            'description': 'Very sensitive (good for distant homologs)',
            'params': {
                's': 8.0,
                'e': 0.01,
                'max_seqs': 1000,
            }
        },
    }

    @classmethod
    def list_presets(cls) -> pd.DataFrame:
        """List available search presets"""
        preset_info = []
        
        for name, config in cls.PRESETS.items():
            param_summary = ', '.join(f"{k}={v}" for k, v in config['params'].items())
            preset_info.append({
                'preset': name,
                'description': config['description'],
                'parameters': param_summary
            })
        
        return pd.DataFrame(preset_info)
    
    def list_databases(self) -> pd.DataFrame:
        """List available databases on the remote server"""
        db_info = []
        
        for alias, db_name in self.AVAILABLE_DBS.items():
            db_path = f"{self.DB_BASE_PATH}/{db_name}"
            
            # Check if database exists and get size
            result = self.conn.run(
                f'if [ -e {db_path} ]; then du -sh {db_path} 2>/dev/null | cut -f1; else echo "NOT_FOUND"; fi',
                hide=True, warn=True
            )
            
            size = result.stdout.strip()
            exists = size != "NOT_FOUND"
            
            db_info.append({
                'alias': alias,
                'database_name': db_name,
                'path': db_path,
                'exists': exists,
                'size': size if exists else 'N/A'
            })
        
        return pd.DataFrame(db_info).sort_values('alias')
    
    def _resolve_database(self, database: str) -> str:
        """Resolve database alias to full path"""
        if database in self.AVAILABLE_DBS:
            db_file = self.AVAILABLE_DBS[database]
            return f"{self.DB_BASE_PATH}/{db_file}"
        
        if database.startswith('/'):
            return database
        
        return f"{self.DB_BASE_PATH}/{database}"
    
    def submit(self, 
           query_file: str,
           database: str,
           job_name: Optional[str] = None,
           preset: Optional[str] = None,
           **mmseqs_params) -> str:
        """Submit a new MMseqs2 search job"""
        
        job_id = str(uuid.uuid4())[:8]
        
        # CREATE LOCAL PROJECT FIRST - ADD THIS BLOCK
        project_dir = self.create_project(
            job_id=job_id,
            job_type='search',
            name=job_name,
            query_file=query_file
        )
        print(f"📁 Created project: {project_dir.name}")
        
        # Rest of the method stays the same...
        job_name = job_name or f"search_{job_id}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"
        
        # Apply preset if specified
        final_params = {}
        if preset:
            if preset not in self.PRESETS:
                available = ', '.join(self.PRESETS.keys())
                raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
            
            final_params = self.PRESETS[preset]['params'].copy()
            print(f"Using preset: {preset} - {self.PRESETS[preset]['description']}")
        
        # User params override preset
        final_params.update(mmseqs_params)
        
        # Resolve database path
        db_path = self._resolve_database(database)
        
        # Verify database exists
        db_check = self.conn.run(
            f'[ -e {db_path} ] && echo "EXISTS" || echo "NOT_FOUND"',
            hide=True, warn=True
        )
        
        if db_check.stdout.strip() == "NOT_FOUND":
            available = ", ".join(self.AVAILABLE_DBS.keys())
            raise ValueError(
                f"Database '{database}' not found at {db_path}\n"
                f"Available aliases: {available}\n"
                f"Or provide a full path to a database."
            )
        
        # Create remote job directory
        self.conn.run(f'mkdir -p {remote_job_path}/tmp', hide=True)
        print(f"Created remote directory: {remote_job_path}")
        
        # Upload query file
        remote_query = f"{remote_job_path}/query.fasta"
        print(f"Uploading query file...")
        self.conn.put(query_file, remote_query)
        
        # Format MMseqs2 parameters (single-letter params use single dash)
        param_str = []
        for k, v in final_params.items():
            param_name = k.replace("_", "-")
            prefix = "-" if len(k) == 1 else "--"
            param_str.append(f'{prefix}{param_name} {v}')
        param_str = ' '.join(param_str)
        
        # Create job script using mmseqs search workflow (not easy-search)
        # This preserves the resultDB for later extraction
        job_script = f"""#!/bin/bash
    set -e

    # Job metadata
    echo "Job started: $(date)" > {remote_job_path}/status.txt
    echo 'RUNNING' >> {remote_job_path}/status.txt

    # Create query database
    mmseqs createdb {remote_query} {remote_job_path}/queryDB --dbtype 1

    # Run MMseqs2 search
    mmseqs search \\
    {remote_job_path}/queryDB \\
    {db_path} \\
    {remote_job_path}/resultDB \\
    {remote_job_path}/tmp \\
    {param_str} \\
    2>&1 | tee {remote_job_path}/mmseqs.log

    # Convert to m8 format for easy viewing
    mmseqs convertalis \\
    {remote_job_path}/queryDB \\
    {db_path} \\
    {remote_job_path}/resultDB \\
    {remote_job_path}/results.m8 \\
    2>&1 | tee -a {remote_job_path}/mmseqs.log

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Job completed: $(date)" >> {remote_job_path}/status.txt
        echo "COMPLETED" >> {remote_job_path}/status.txt
    else
        echo "Job failed: $(date)" >> {remote_job_path}/status.txt
        echo "FAILED" >> {remote_job_path}/status.txt
    fi

    # Cleanup tmp files but keep resultDB
    rm -rf {remote_job_path}/tmp
    """
        
        # Upload and execute job script
        script_path = f"{remote_job_path}/run.sh"
        self.conn.put(
            local=self._write_temp_script(job_script),
            remote=script_path
        )
        self.conn.run(f'chmod +x {script_path}', hide=True)
        
        # Submit job in background with nohup
        result = self.conn.run(
            f'nohup {script_path} > {remote_job_path}/nohup.out 2>&1 & echo $!',
            hide=True
        )
        pid = result.stdout.strip()
        
        # Save PID
        self.conn.run(f'echo {pid} > {remote_job_path}/pid.txt', hide=True)
        
        # Update local job database
        job_db = self._load_job_db()
        job_db[job_id] = {
            'job_type': 'search',
            'name': job_name,
            'database': database,
            'database_path': db_path,
            'query_file': str(query_file),
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid,
            'preset': preset,
            'parameters': final_params
        }
        self._save_job_db(job_db)
        
        print(f"✓ Job submitted: {job_name} (ID: {job_id})")
        print(f"  Database: {database} ({db_path})")
        print(f"  PID: {pid}")
        
        return job_id
        
    def _parse_mmseqs_progress(self, log_content: str) -> Dict:
        """Parse MMseqs2 log for progress information"""
        progress = {
            'current_step': None,
            'prefilter_step': None,
            'total_prefilter_steps': None,
            'current_operation': None,
        }
        
        if not log_content or log_content == "No log file":
            return progress
        
        lines = log_content.strip().split('\n')
        
        # Parse prefiltering step from most recent line
        prefilter_pattern = r'Process prefiltering step (\d+) of (\d+)'
        for line in reversed(lines):
            match = re.search(prefilter_pattern, line)
            if match:
                progress['prefilter_step'] = int(match.group(1))
                progress['total_prefilter_steps'] = int(match.group(2))
                progress['current_step'] = 'prefilter'
                break
        
        # Determine current operation
        operations = [
            ('Index table: counting k-mers', 'Counting k-mers'),
            ('Index table: fill', 'Building index'),
            ('Starting prefiltering scores', 'Computing scores'),
            ('align', 'Aligning'),
            ('convertalis', 'Finalizing'),
        ]
        
        for line in reversed(lines[:50]):  # Check recent lines only
            for keyword, operation in operations:
                if keyword in line:
                    progress['current_operation'] = operation
                    return progress
        
        return progress
    
    def detailed_status(self, job_id: str) -> Dict:
        """Get detailed status with progress parsing"""
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
        
        # Determine status
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
        
        # Parse progress if running
        progress = {}
        if status in ['RUNNING', 'SUBMITTED']:
            log_result = self.conn.run(
                f'tail -n 100 {remote_path}/mmseqs.log 2>/dev/null || echo "No log file"',
                hide=True, warn=True
            )
            progress = self._parse_mmseqs_progress(log_result.stdout)
        
        # Update local DB
        job_info['status'] = status
        job_info['last_checked'] = datetime.now().isoformat()
        self._save_job_db(job_db)
        
        return {
            'job_id': job_id,
            'name': job_info['name'],
            'status': status,
            'runtime': runtime,
            'job_type': job_info.get('job_type', 'search'),
            **progress
        }
    
    def print_detailed_status(self, job_id: str, watch: bool = False, animation_frame: int = 0):
        """Print formatted status with optional live updates"""
        from itertools import cycle
        
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Clear in watch mode (for notebooks)
        if watch and animation_frame > 0:
            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except ImportError:
                print('\033[2J\033[H')  # Clear terminal
        
        status = self.detailed_status(job_id)
        
        print(f"\n{'='*60}")
        print(f"Job: {status['name']} ({status['job_id']})")
        print(f"Status: {status['status']} | Runtime: {status['runtime']}")
        print(f"{'='*60}")
        
        if status['status'] in ['RUNNING', 'SUBMITTED']:
            # Show progress
            if status.get('prefilter_step') and status.get('total_prefilter_steps'):
                total = status['total_prefilter_steps']
                current = status['prefilter_step']
                pct = (current / total) * 100
                
                # Progress bar
                bar_length = 40
                filled = int(bar_length * current / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                icon = spinner[animation_frame % len(spinner)] if watch else '⟳'
                print(f"\n{icon} Prefilter: Step {current}/{total} ({pct:.1f}%)")
                print(f"  [{bar}]")
            
            if status.get('current_operation'):
                print(f"\n🔄 {status['current_operation']}")
            
            if watch:
                print(f"\n(Press Ctrl+C to stop watching)")
        
        print(f"{'='*60}\n")
        
        # Watch mode
        if watch and status['status'] in ['RUNNING', 'SUBMITTED']:
            try:
                import time
                time.sleep(1)
                self.print_detailed_status(job_id, watch=True, animation_frame=animation_frame + 1)
            except KeyboardInterrupt:
                print("\nStopped watching.")
    
    def status(self, job_id: str, verbose: bool = False) -> Dict:
        """
        Check job status
        
        Args:
            job_id: Job identifier
            verbose: If True, return detailed status with progress
        """
        if verbose:
            return self.detailed_status(job_id)
        
        # Simple status (keeps original behavior)
        job_db = self._load_job_db()
        
        if job_id not in job_db:
            return {'status': 'UNKNOWN', 'error': 'Job ID not found'}
        
        job_info = job_db[job_id]
        remote_path = job_info['remote_path']
        pid = job_info['pid']
        
        # Check process
        ps_result = self.conn.run(f'ps -p {pid} -o pid=', warn=True, hide=True)
        is_running = ps_result.ok
        
        # Read status file
        status_result = self.conn.run(
            f'cat {remote_path}/status.txt 2>/dev/null || echo "NO_STATUS"',
            hide=True, warn=True
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
            'job_type': job_info.get('job_type', 'search')
        }
    
    def wait(self, job_id: str, check_interval: int = 30, verbose: bool = True, show_progress: bool = False):
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
        
        # Simple waiting
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
    
    def list_jobs(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """
        List all search jobs, optionally filtered by status
        
        Args:
            status_filter: Optional status to filter by (RUNNING, COMPLETED, etc.)
        
        Returns:
            DataFrame with job information
        """
        job_db = self._load_job_db()
        
        # Filter for search jobs only
        search_jobs = {k: v for k, v in job_db.items() if v.get('job_type') == 'search'}
        
        if not search_jobs:
            return pd.DataFrame()
        
        jobs_list = []
        for job_id, info in search_jobs.items():
            current_status = self.status(job_id)
            
            if status_filter and current_status['status'] != status_filter:
                continue
            
            jobs_list.append({
                'job_id': job_id,
                'name': info['name'],
                'status': current_status['status'],
                'database': info['database'],
                'submitted': info['submitted_at'],
                'runtime': current_status['runtime']
            })
        
        df = pd.DataFrame(jobs_list)
        if not df.empty:
            df = df.sort_values('submitted', ascending=False)
        
        return df
    
    def get_results(self, 
                job_id: str, 
                parse: bool = True,
                download_sequences: bool = False) -> Union[Path, pd.DataFrame, Dict]:
        """
        Download and optionally parse job results
        
        Results are automatically saved to the job's project directory.
        
        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame; if False, return Path to m8 file
            download_sequences: Also download hit sequences as FASTA
        
        Returns:
            DataFrame (if parse=True), Path (if parse=False), or Dict (if download_sequences=True)
        """
        # Check job status
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")
        
        # Get project directory
        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            # Shouldn't happen, but create if missing
            project_dir = self.create_project(job_id, 'search')
        
        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']
        
        # Download m8 results
        m8_file = project_dir / "results.m8"
        if not m8_file.exists():
            print(f"Downloading results...")
            self.conn.get(f"{remote_path}/results.m8", str(m8_file))
            print(f"✓ Downloaded to {project_dir.name}/results.m8")
        
        # Download log
        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(f"{remote_path}/mmseqs.log", str(log_file))
        
        # Download sequences if requested
        if download_sequences:
            fasta_file = project_dir / "hits.fasta"
            if not fasta_file.exists():
                print(f"Extracting hit sequences...")
                db_path = job_db[job_id]['database_path']
                remote_fasta = f"{remote_path}/hits.fasta"
                
                cmd = f"""
set -e
mmseqs createseqfiledb {db_path} {remote_path}/resultDB {remote_path}/fastaDB
mmseqs result2flat {db_path} {db_path} {remote_path}/fastaDB {remote_fasta} --use-fasta-header
rm -f {remote_path}/fastaDB*
"""
                result = self.conn.run(cmd, warn=True, hide=False)
                if result.failed:
                    raise RuntimeError("Failed to extract sequences")
                
                self.conn.get(remote_fasta, str(fasta_file))
                
                with open(fasta_file) as f:
                    n_seqs = sum(1 for line in f if line.startswith('>'))
                print(f"✓ Extracted {n_seqs} sequences to {project_dir.name}/hits.fasta")
        
        # Parse if requested
        if parse:
            columns = [
                'query', 'target', 'identity', 'alignment_length',
                'mismatches', 'gap_openings', 'q_start', 'q_end',
                't_start', 't_end', 'evalue', 'bit_score'
            ]
            df = pd.read_csv(m8_file, sep='\t', names=columns, comment='#')
            print(f"✓ Parsed {len(df)} alignments")
            
            if download_sequences:
                return {'dataframe': df, 'm8': m8_file, 'fasta': fasta_file}
            return df
        
        if download_sequences:
            return {'m8': m8_file, 'fasta': fasta_file}
        
        return m8_file

    def download(self, job_id: str, local_dir: str = '.') -> Path:
        """
        Download job results (m8 format)
        
        DEPRECATED: Use get_results(job_id, format='m8') instead
        
        Args:
            job_id: Job identifier
            local_dir: Local directory to save results
        
        Returns:
            Path to downloaded results file
        """
        import warnings
        warnings.warn(
            "download() is deprecated. Use get_results(job_id, format='m8') instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_results(job_id, output_dir=local_dir, format='m8', download_log=True)

    def get_hit_sequences(self, job_id: str, output_fasta: Optional[str] = None) -> Path:
        """
        Extract hit sequences from search results as FASTA
        
        DEPRECATED: Use get_results(job_id, format='fasta') instead
        
        Args:
            job_id: Job identifier
            output_fasta: Optional output path
        
        Returns:
            Path to FASTA file with hit sequences
        """
        import warnings
        warnings.warn(
            "get_hit_sequences() is deprecated. Use get_results(job_id, format='fasta') instead",
            DeprecationWarning,
            stacklevel=2
        )
        
        if output_fasta:
            output_dir = Path(output_fasta).parent
            fasta_file = self.get_results(job_id, output_dir=str(output_dir), format='fasta')
            # Rename to requested name
            Path(fasta_file).rename(output_fasta)
            return Path(output_fasta)
        else:
            return self.get_results(job_id, format='fasta')