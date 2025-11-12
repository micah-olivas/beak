from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import time

from .base import RemoteJobManager


class ClustalAlign(RemoteJobManager):
    """Clustal Omega alignment manager for remote execution"""
    
    def submit(self, 
               input_file: str,
               job_name: Optional[str] = None,
               output_format: str = "fasta",
               **clustal_params) -> str:
        """
        Submit a new Clustal Omega alignment job
        
        Args:
            input_file: Path to local FASTA file with sequences to align
            job_name: Optional human-readable job name
            output_format: Output format (fasta, clustal, msf, phylip, selex, stockholm, vienna)
            **clustal_params: Clustal Omega parameters (e.g., threads=4, iterations=5)
        
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:8]
        job_name = job_name or f"align_{job_id}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"
        
        # Create remote job directory
        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)
        print(f"Created remote directory: {remote_job_path}")
        
        # Upload input file
        remote_input = f"{remote_job_path}/input.fasta"
        print(f"Uploading input file...")
        self.conn.put(input_file, remote_input)
        
        # Format Clustal Omega parameters
        param_str = []
        
        # Add output format if not fasta
        if output_format != "fasta":
            param_str.append(f'--outfmt={output_format}')
        
        # Add other parameters
        for k, v in clustal_params.items():
            param_name = k.replace("_", "-")
            # Clustal Omega uses double dash for all options
            param_str.append(f'--{param_name}={v}')
        
        param_str = ' '.join(param_str)
        
        # Create job script
        job_script = f"""#!/bin/bash
set -e

# Job metadata
echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

# Run Clustal Omega
clustalo \\
  -i {remote_input} \\
  -o {remote_job_path}/alignment.{output_format} \\
  {param_str} \\
  2>&1 | tee {remote_job_path}/clustalo.log

# Check if successful
if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi
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
            'job_type': 'align',
            'name': job_name,
            'input_file': str(input_file),
            'output_format': output_format,
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid,
            'parameters': clustal_params
        }
        self._save_job_db(job_db)
        
        print(f"✓ Job submitted: {job_name} (ID: {job_id})")
        print(f"  Output format: {output_format}")
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
        Download alignment results
        
        Args:
            job_id: Job identifier
            local_dir: Local directory to save results
        
        Returns:
            Path to downloaded alignment file
        """
        status_info = self.status(job_id)
        
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")
        
        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']
        output_format = job_db[job_id]['output_format']
        
        # Create local output directory
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download alignment
        local_file = local_dir / f"{job_id}_alignment.{output_format}"
        self.conn.get(
            remote=f"{remote_path}/alignment.{output_format}",
            local=str(local_file)
        )
        
        # Also download log
        local_log = local_dir / f"{job_id}_log.txt"
        self.conn.get(
            remote=f"{remote_path}/clustalo.log",
            local=str(local_log)
        )
        
        print(f"✓ Downloaded alignment to {local_file}")
        print(f"✓ Downloaded log to {local_log}")
        
        return local_file
    
    def list_jobs(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """
        List all alignment jobs, optionally filtered by status
        
        Args:
            status_filter: Optional status to filter by (RUNNING, COMPLETED, etc.)
        
        Returns:
            DataFrame with job information
        """
        job_db = self._load_job_db()
        
        # Filter for alignment jobs only
        align_jobs = {k: v for k, v in job_db.items() if v.get('job_type') == 'align'}
        
        if not align_jobs:
            return pd.DataFrame()
        
        jobs_list = []
        for job_id, info in align_jobs.items():
            current_status = self.status(job_id)
            
            if status_filter and current_status['status'] != status_filter:
                continue
            
            jobs_list.append({
                'job_id': job_id,
                'name': info['name'],
                'status': current_status['status'],
                'output_format': info.get('output_format', 'fasta'),
                'submitted': info['submitted_at'],
                'runtime': current_status['runtime']
            })
        
        df = pd.DataFrame(jobs_list)
        if not df.empty:
            df = df.sort_values('submitted', ascending=False)
        
        return df
    
    def get_results(self, job_id: str, parse: bool = False):
        """
        Download and optionally parse alignment results
        
        Args:
            job_id: Job identifier
            parse: If True, return BioPython alignment object (requires output_format='fasta')
        
        Returns:
            Path to alignment file, or BioPython MultipleSeqAlignment if parse=True
        """
        results_file = self.download(job_id)
        
        if not parse:
            return results_file
        
        # Parse alignment with BioPython
        job_db = self._load_job_db()
        output_format = job_db[job_id]['output_format']
        
        if output_format != 'fasta':
            print(f"Warning: Parsing only supported for FASTA format, got {output_format}")
            return results_file
        
        from Bio import AlignIO
        
        alignment = AlignIO.read(results_file, "fasta")
        
        print(f"✓ Parsed alignment: {len(alignment)} sequences, {alignment.get_alignment_length()} positions")
        
        return alignment