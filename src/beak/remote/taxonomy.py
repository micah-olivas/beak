import re
import uuid
import time
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from .base import RemoteJobManager

class MMseqsTaxonomy(RemoteJobManager):
    """MMseqs2 taxonomy assignment for remote execution"""
    
    # Available taxonomy databases
    AVAILABLE_TAX_DBS = {
        'uniref90': 'UniRef90',
        'uniref100': 'UniRef100',
        'uniref50': 'uniref50',
        'uniprotkb': 'UniProtKB',
        'swissprot': 'swissprot',
        'trembl': 'TrEMBL',
    }
    
    def list_databases(self) -> pd.DataFrame:
        """List available taxonomy databases on the remote server"""
        db_info = []
        
        for alias, db_name in self.AVAILABLE_TAX_DBS.items():
            db_path = f"{self.DB_BASE_PATH}/{db_name}"
            
            # Check if database exists and has taxonomy files
            result = self.conn.run(
                f'if [ -e {db_path} ]; then du -sh {db_path} 2>/dev/null | cut -f1; else echo "NOT_FOUND"; fi',
                hide=True, warn=True
            )
            
            size = result.stdout.strip()
            exists = size != "NOT_FOUND"
            
            # Check for taxonomy files
            tax_check = self.conn.run(
                f'[ -e {db_path}_taxonomy ] && echo "YES" || echo "NO"',
                hide=True, warn=True
            )
            has_taxonomy = tax_check.stdout.strip() == "YES"
            
            db_info.append({
                'alias': alias,
                'database_name': db_name,
                'path': db_path,
                'exists': exists,
                'has_taxonomy': has_taxonomy,
                'size': size if exists else 'N/A'
            })
        
        return pd.DataFrame(db_info).sort_values('alias')
    
    def _resolve_database(self, database: str) -> str:
        """Resolve database alias to full path"""
        if database in self.AVAILABLE_TAX_DBS:
            db_file = self.AVAILABLE_TAX_DBS[database]
            return f"{self.DB_BASE_PATH}/{db_file}"
        
        if database.startswith('/'):
            return database
        
        return f"{self.DB_BASE_PATH}/{database}"
    
    def submit(self, 
           query_file: str,
           # default to UniProtKB
           database: str = 'uniprotkb',
           job_name: Optional[str] = None,
           tax_lineage: bool = True,
           **mmseqs_params) -> str:
        """
        Submit a new MMseqs2 taxonomy assignment job
        
        Args:
            query_file: Path to local FASTA file
            database: Database alias (e.g., 'uniref90') or full path
            job_name: Optional human-readable job name
            tax_lineage: If True, include full taxonomic lineage (--tax-lineage 1)
            **mmseqs_params: Additional MMseqs2 taxonomy parameters
        
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:8]
        job_name = job_name or f"taxonomy_{job_id}"
        
        # CREATE LOCAL PROJECT FIRST
        project_dir = self.create_project(
            job_id=job_id,
            job_type='taxonomy',
            name=job_name,
            query_file=query_file
        )
        print(f"📁 Created project: {project_dir.name}")
        
        remote_job_path = f"{self.remote_job_dir}/{job_id}"
        
        # Resolve database path
        db_path = self._resolve_database(database)
        
        # Verify database exists
        db_check = self.conn.run(
            f'[ -e {db_path} ] && echo "EXISTS" || echo "NOT_FOUND"',
            hide=True, warn=True
        )
        
        if db_check.stdout.strip() == "NOT_FOUND":
            available = ", ".join(self.AVAILABLE_TAX_DBS.keys())
            raise ValueError(
                f"Database '{database}' not found at {db_path}\n"
                f"Available aliases: {available}\n"
                f"Or provide a full path to a database."
            )
        
        # Check if taxonomy files exist
        tax_check = self.conn.run(
            f'[ -e {db_path}_taxonomy ] && echo "EXISTS" || echo "NOT_FOUND"',
            hide=True, warn=True
        )
        
        if tax_check.stdout.strip() == "NOT_FOUND":
            raise ValueError(
                f"Database '{database}' does not have taxonomy information.\n"
                f"Missing: {db_path}_taxonomy"
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
        
        # Add tax-lineage parameter
        if tax_lineage:
            param_str.append('--tax-lineage 1')
        
        # Add other parameters
        for k, v in mmseqs_params.items():
            param_name = k.replace("_", "-")
            prefix = "-" if len(k) == 1 else "--"
            param_str.append(f'{prefix}{param_name} {v}')
        
        param_str = ' '.join(param_str)
        
        # Create job script
        job_script = f"""#!/bin/bash
set -e

# Job metadata
echo "Job started: $(date)" > {remote_job_path}/status.txt
echo "RUNNING" >> {remote_job_path}/status.txt

# Create query database
mmseqs createdb {remote_query} {remote_job_path}/queryDB \\
  2>&1 | tee {remote_job_path}/mmseqs.log

# Run taxonomy assignment
mmseqs taxonomy \\
  {remote_job_path}/queryDB \\
  {db_path} \\
  {remote_job_path}/taxonomyResult \\
  {remote_job_path}/tmp \\
  {param_str} \\
  2>&1 | tee -a {remote_job_path}/mmseqs.log

# Convert to TSV
mmseqs createtsv \\
  {remote_job_path}/queryDB \\
  {remote_job_path}/taxonomyResult \\
  {remote_job_path}/results.tsv \\
  2>&1 | tee -a {remote_job_path}/mmseqs.log

# Check if successful
if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi

# Cleanup tmp files
rm -rf {remote_job_path}/tmp {remote_job_path}/queryDB* {remote_job_path}/taxonomyResult*
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
            'job_type': 'taxonomy',
            'name': job_name,
            'database': database,
            'database_path': db_path,
            'query_file': str(query_file),
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid,
            'parameters': {**mmseqs_params, 'tax_lineage': tax_lineage}
        }
        self._save_job_db(job_db)
        
        print(f"✓ Job submitted: {job_name} (ID: {job_id})")
        print(f"  Database: {database} ({db_path})")
        print(f"  PID: {pid}")
        
        return job_id
    
    def wait(self, job_id: str, check_interval: int = 30, verbose: bool = True):
        """
        Wait for job to complete with progress updates
        
        Args:
            job_id: Job identifier
            check_interval: Seconds between status checks
            verbose: Print progress updates
        """
        if verbose:
            print(f"Waiting for job {job_id}...")
        
        while True:
            status_info = self.status(job_id)
            
            if verbose:
                print(f"  [{status_info['runtime']}] Status: {status_info['status']}")
            
            if status_info['status'] in ['COMPLETED', 'FAILED', 'UNKNOWN', 'CANCELLED']:
                break
            
            time.sleep(check_interval)
        
        if verbose:
            if status_info['status'] == 'COMPLETED':
                print(f"✓ Job completed in {status_info['runtime']}")
            else:
                print(f"✗ Job {status_info['status'].lower()}")
        
        return status_info['status']
    
    def annotate_search_results(self, 
                           hits_fasta: Path,
                           database: str = 'uniprotkb',
                           job_name: Optional[str] = None,
                           tax_lineage: bool = True,
                           **mmseqs_params) -> str:
        """
        Perform taxonomy annotation on a FASTA file (e.g., from search results)
        
        Args:
            hits_fasta: Path to FASTA file with sequences to annotate
            database: Database alias (e.g., 'uniprotkb') or full path
            job_name: Optional human-readable job name
            tax_lineage: If True, include full taxonomic lineage (--tax-lineage 1)
            **mmseqs_params: Additional MMseqs2 taxonomy parameters
        
        Returns:
            job_id: Unique job identifier
        """
        return self.submit(
            query_file=str(hits_fasta),
            database=database,
            job_name=job_name,
            tax_lineage=tax_lineage,
            **mmseqs_params
        )
    
    def _parse_mmseqs_progress(self, log_content: str) -> Dict:
        """Parse MMseqs2 taxonomy log for progress information"""
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
        
        # Determine current operation from taxonomy workflow
        operations = [
            ('Index table: counting k-mers', 'Counting k-mers'),
            ('Index table: fill', 'Building index'),
            ('Starting prefiltering scores', 'Computing scores'),
            ('align', 'Aligning'),
            ('lca', 'Computing LCA'),
            ('createtsv', 'Generating results'),
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
            'job_type': job_info.get('job_type', 'taxonomy'),
            **progress
        }

    def print_detailed_status(self, job_id: str, watch: bool = False, animation_frame: int = 0):
        """Print formatted status with optional live updates"""
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
        print(f"Type: Taxonomy | Status: {status['status']} | Runtime: {status['runtime']}")
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
        
        # Simple status (keep existing implementation but add last_checked update)
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
            'job_type': job_info.get('job_type', 'taxonomy')
        }
    
    def list_jobs(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """
        List all taxonomy jobs, optionally filtered by status
        
        Args:
            status_filter: Optional status to filter by (RUNNING, COMPLETED, etc.)
        
        Returns:
            DataFrame with job information
        """
        job_db = self._load_job_db()
        
        # Filter for taxonomy jobs only
        taxonomy_jobs = {k: v for k, v in job_db.items() if v.get('job_type') == 'taxonomy'}
        
        if not taxonomy_jobs:
            return pd.DataFrame()
        
        jobs_list = []
        for job_id, info in taxonomy_jobs.items():
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
    
    def parse_taxonomy_lineage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the taxonomic lineage string into separate columns
        
        Args:
            df: DataFrame from get_results() with lineage column
        
        Returns:
            DataFrame with additional parsed taxonomy columns
        """
        if 'lineage' not in df.columns:
            print("No lineage column found. Run with --tax-lineage 1")
            return df
        
        # Parse lineage strings
        def parse_lineage(lineage_str):
            """Parse lineage string into dict of taxonomic ranks"""
            if pd.isna(lineage_str) or lineage_str == '':
                return {}
            
            ranks = {
                '-': 'unranked',
                'd': 'superkingdom',  # Rarely used
                'k': 'domain',        # This is actually domain (Bacteria, Archaea, Eukaryota)
                'p': 'phylum',
                'c': 'class',
                'o': 'order',
                'f': 'family',
                'g': 'genus',
                's': 'species'
            }
            
            taxonomy = {}
            parts = lineage_str.split(';')
            
            # First pass: look for domain in common names
            full_lineage = lineage_str.lower()
            if 'bacteria' in full_lineage and 'archaea' not in full_lineage:
                taxonomy['domain'] = 'Bacteria'
            elif 'archaea' in full_lineage:
                taxonomy['domain'] = 'Archaea'
            elif 'eukaryota' in full_lineage:
                taxonomy['domain'] = 'Eukaryota'
            
            # Second pass: parse all ranks
            for part in parts:
                part = part.strip()
                if '_' in part:
                    rank_code, name = part.split('_', 1)
                    
                    # Skip empty names
                    if not name or name == '':
                        continue
                    
                    # Map rank code to rank name
                    if rank_code == 'k':
                        # k_ prefix is used for kingdom/domain level
                        # If we haven't found domain yet, use this
                        if 'domain' not in taxonomy:
                            taxonomy['domain'] = name
                        # Also save as kingdom for compatibility
                        taxonomy['kingdom'] = name
                    elif rank_code in ranks:
                        rank_name = ranks[rank_code]
                        if rank_name != 'unranked':
                            taxonomy[rank_name] = name
            
            return taxonomy
        
        # Apply parsing
        lineage_data = df['lineage'].apply(parse_lineage)
        
        # Create columns for each rank
        rank_columns = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for rank in rank_columns:
            df[rank] = lineage_data.apply(lambda x: x.get(rank, None))
        
        return df

    def get_results(self, job_id: str, parse: bool = True, parse_lineage: bool = True):
        """
        Download and optionally parse taxonomy results
        
        Results are automatically saved to the job's project directory.
        
        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame
            parse_lineage: If True and parse=True, parse lineage into separate columns
        
        Returns:
            Path or DataFrame depending on parse parameter
        """
        # Check job status
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")
        
        # Get project directory
        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            # Shouldn't happen, but create if missing
            project_dir = self.create_project(job_id, 'taxonomy')
        
        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']
        
        # Download results to project directory
        results_file = project_dir / "taxonomy_results.tsv"
        if not results_file.exists():
            print(f"Downloading results...")
            self.conn.get(
                remote=f"{remote_path}/results.tsv",
                local=str(results_file)
            )
            print(f"✓ Downloaded to {project_dir.name}/taxonomy_results.tsv")
        
        # Download log
        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(
                remote=f"{remote_path}/mmseqs.log",
                local=str(log_file)
            )
        
        if not parse:
            return results_file
        
        # Parse taxonomy TSV
        df = pd.read_csv(results_file, sep='\t', header=None)
        
        # Determine column names based on number of columns
        if len(df.columns) == 5:
            df.columns = ['query', 'taxid', 'rank', 'scientific_name', 'lineage']
        elif len(df.columns) == 4:
            df.columns = ['query', 'taxid', 'rank', 'scientific_name']
        else:
            # Generic column names if unexpected format
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
        
        print(f"✓ Parsed taxonomy for {len(df)} sequences")
        
        # Parse lineage if requested and available
        if parse_lineage and 'lineage' in df.columns:
            df = self.parse_taxonomy_lineage(df)
            print(f"✓ Parsed taxonomic lineage into ranks")
        
        return df