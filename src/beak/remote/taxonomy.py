from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import time

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
    
    def download(self, job_id: str, local_dir: str = '.') -> Path:
        """
        Download taxonomy results
        
        Args:
            job_id: Job identifier
            local_dir: Local directory to save results
        
        Returns:
            Path to downloaded results file
        """
        status_info = self.status(job_id)
        
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")
        
        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']
        
        # Create local output directory
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download results
        local_file = local_dir / f"{job_id}_taxonomy.tsv"
        self.conn.get(
            remote=f"{remote_path}/results.tsv",
            local=str(local_file)
        )
        
        # Also download log
        local_log = local_dir / f"{job_id}_log.txt"
        self.conn.get(
            remote=f"{remote_path}/mmseqs.log",
            local=str(local_log)
        )
        
        print(f"✓ Downloaded results to {local_file}")
        print(f"✓ Downloaded log to {local_log}")
        
        return local_file
    
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
    
    def get_results(self, job_id: str, parse: bool = True):
        """
        Download and optionally parse taxonomy results
        
        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame
        
        Returns:
            Path or DataFrame depending on parse parameter
        """
        results_file = self.download(job_id)
        
        if not parse:
            return results_file
        
        # Parse taxonomy TSV
        # Standard columns: query, taxid, rank, scientific_name, [lineage if --tax-lineage 1]
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
        
        return df