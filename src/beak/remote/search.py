from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import time

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
                'max_seqs': 10000,  # allow many results
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
        """
        Submit a new MMseqs2 search job
        
        Args:
            query_file: Path to local FASTA file
            database: Database alias (e.g., 'uniref90') or full path
            job_name: Optional human-readable job name
            preset: Search preset ('default', 'fast', 'sensitive', 'exhaustive', 'very_sensitive')
            **mmseqs_params: MMseqs2 parameters (override preset values)
        
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:8]
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
        
        # Create job script
        job_script = f"""#!/bin/bash
set -e

# Job metadata
echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

# Run MMseqs2
mmseqs easy-search \\
{remote_query} \\
{db_path} \\
{remote_job_path}/results.m8 \\
{remote_job_path}/tmp \\
{param_str} \\
2>&1 | tee {remote_job_path}/mmseqs.log

# Check if successful
if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi

# Cleanup tmp files
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
        Download job results
        
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
        local_file = local_dir / f"{job_id}_results.m8"
        self.conn.get(
            remote=f"{remote_path}/results.m8",
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
    
    def get_results(self, job_id: str, parse: bool = True):
        """
        Download and optionally parse results
        
        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame
        
        Returns:
            Path or DataFrame depending on parse parameter
        """
        results_file = self.download(job_id)
        
        if not parse:
            return results_file
        
        # Parse MMseqs2 output (m8 format)
        columns = [
            'query', 'target', 'identity', 'alignment_length',
            'mismatches', 'gap_openings', 'q_start', 'q_end',
            't_start', 't_end', 'evalue', 'bit_score'
        ]
        
        df = pd.read_csv(results_file, sep='\t', names=columns, comment='#')
        
        print(f"✓ Parsed {len(df)} alignments")
        
        return df
    
    def get_hit_sequences(self, job_id: str, output_fasta: Optional[str] = None) -> Path:
        """
        Extract hit sequences from search results as FASTA
        
        Args:
            job_id: Job identifier
            output_fasta: Optional output path, defaults to {job_id}_hits.fasta
        
        Returns:
            Path to FASTA file with hit sequences
        """
        job_db = self._load_job_db()
        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")
        
        remote_path = job_db[job_id]['remote_path']
        db_path = job_db[job_id]['database_path']
        remote_output = f"{remote_path}/hits.fasta"
        
        # Use mmseqs result2profile to extract sequences directly from m8
        # This is much faster than grep on large lookup files
        cmd = f"""
        # Convert m8 to mmseqs result format and extract target sequences
        # Create a query DB (needed for result2profile)
        mmseqs createdb {remote_path}/query.fasta {remote_path}/queryDB
        
        # Convert blast-tab (m8) to mmseqs result database
        mmseqs convertalis {remote_path}/queryDB {db_path} {remote_path}/results.m8 {remote_path}/resultDB --format-mode 4
        
        # Extract unique target sequences
        mmseqs result2profile {remote_path}/queryDB {db_path} {remote_path}/resultDB {remote_path}/profileDB
        mmseqs profile2seq {remote_path}/profileDB {remote_path}/hitDB
        
        # Or simpler: just use result2flat to get unique hits
        mmseqs result2flat {remote_path}/queryDB {db_path} {remote_path}/resultDB {remote_output} --use-fasta-header
        
        # Cleanup
        rm -f {remote_path}/queryDB* {remote_path}/resultDB* {remote_path}/profileDB* {remote_path}/hitDB*
        """
        
        result = self.conn.run(cmd, warn=True)
        
        if result.failed:
            raise RuntimeError(f"Failed to extract sequences")
        
        output_fasta = output_fasta or f"{job_id}_hits.fasta"
        self.conn.get(remote_output, output_fasta)
        
        with open(output_fasta) as f:
            n_seqs = sum(1 for line in f if line.startswith('>'))
        
        print(f"✓ Extracted {n_seqs} hit sequences to {output_fasta}")
        return Path(output_fasta)
