from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import time

from .base import RemoteJobManager


class ESMEmbeddings(RemoteJobManager):
    """ESM protein language model embeddings via Docker"""
    
    # Available ESM models (filtered for 11GB VRAM)
    AVAILABLE_MODELS = {
        'esm2_t6_8M_UR50D': {
            'name': 'ESM2-8M',
            'params': '8M',
            'vram_gb': 0.1,
            'description': 'Tiny model, very fast'
        },
        'esm2_t12_35M_UR50D': {
            'name': 'ESM2-35M',
            'params': '35M',
            'vram_gb': 0.2,
            'description': 'Small model, fast'
        },
        'esm2_t30_150M_UR50D': {
            'name': 'ESM2-150M',
            'params': '150M',
            'vram_gb': 0.6,
            'description': 'Medium model, good balance'
        },
        'esm2_t33_650M_UR50D': {
            'name': 'ESM2-650M',
            'params': '650M',
            'vram_gb': 2.5,
            'description': 'Large model (max for 1080 Ti), best quality'
        },
    }
    
    def __init__(self, host: str, user: str, key_path: Optional[str] = None, 
                 remote_job_dir: Optional[str] = None):
        super().__init__(host, user, key_path, remote_job_dir)
        
        # Verify Docker is available
        self._verify_docker()
        
        # Ensure Docker service is running
        self._ensure_docker_service()
    
    def _verify_docker(self):
        """Verify Docker is available on remote"""
        result = self.conn.run('docker --version', hide=True, warn=True)
        if not result.ok:
            raise RuntimeError(
                "Docker not found on remote server. "
                "Please install Docker or use a different execution mode."
            )
    
    @classmethod
    def list_models(cls) -> pd.DataFrame:
        """List available ESM models"""
        models = []
        for model_id, info in cls.AVAILABLE_MODELS.items():
            models.append({
                'model_id': model_id,
                'name': info['name'],
                'parameters': info['params'],
                'vram_gb': info['vram_gb'],
                'description': info['description']
            })
        return pd.DataFrame(models)
    
    def submit(self,
               input_file: str,
               model: str = 'esm2_t33_650M_UR50D',
               job_name: Optional[str] = None,
               repr_layers: List[int] = [-1],
               include_mean: bool = True,     
               include_per_tok: bool = False, 
               gpu_id: int = 0) -> str:
        """
        Submit ESM embedding generation job
        
        Args:
            input_file: Path to local FASTA file
            model: ESM model ID (use list_models() to see options)
            job_name: Optional human-readable job name
            repr_layers: Which layers to extract ([-1] = last layer)
            include_mean: Include mean pooled embeddings
            include_per_tok: Include per-token embeddings (large files!)
            gpu_id: GPU device ID
        
        Returns:
            job_id: Unique job identifier
        """
        if model not in self.AVAILABLE_MODELS:
            available = ', '.join(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Available: {available}")
        
        job_id = str(uuid.uuid4())[:8]
        job_name = job_name or f"esm_{job_id}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"
        
        # Create remote job directory
        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)
        
        # Upload input file
        remote_input = f"{remote_job_path}/input.fasta"
        self.conn.put(input_file, remote_input)
        
        # Generate job script
        job_script = self._generate_job_script(
            remote_job_path, remote_input, model, repr_layers,
            include_mean, include_per_tok, gpu_id
        )
        
        # Upload and execute
        script_path = f"{remote_job_path}/run.sh"
        self.conn.put(
            local=self._write_temp_script(job_script),
            remote=script_path
        )
        self.conn.run(f'chmod +x {script_path}', hide=True)
        
        # Submit in background
        result = self.conn.run(
            f'nohup {script_path} > {remote_job_path}/nohup.out 2>&1 & echo $!',
            hide=True
        )
        pid = result.stdout.strip()
        
        # Update job database
        job_db = self._load_job_db()
        job_db[job_id] = {
            'job_type': 'embeddings',
            'name': job_name,
            'model': model,
            'input_file': str(input_file),
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid,
            'parameters': {
                'repr_layers': repr_layers,
                'include_mean': include_mean,
                'include_per_tok': include_per_tok,
                'gpu_id': gpu_id
            }
        }
        self._save_job_db(job_db)
        
        model_info = self.AVAILABLE_MODELS[model]
        print(f"✓ Job submitted: {job_name} (ID: {job_id})")
        print(f"  Model: {model_info['name']} ({model_info['params']} parameters)")
        print(f"  GPU: {gpu_id}")
        
        return job_id
    
    def _generate_job_script(self, remote_job_path, remote_input, model,
                        repr_layers, include_mean, include_per_tok, gpu_id):
        """Generate Docker execution script"""
        
        docker_dir = f"{self.remote_job_dir}/docker"
        output_dir = f"{remote_job_path}/embeddings"
        
        # Convert repr_layers to string
        repr_layers_str = ' '.join(map(str, repr_layers))
        
        # Build flags
        mean_flag = "--include-mean" if include_mean else ""
        tok_flag = "--include-per-tok" if include_per_tok else ""
        
        script = f"""#!/bin/bash
set -e

echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

mkdir -p {output_dir}

# Use docker compose exec to run in existing container
cd {docker_dir}
docker compose exec -T embeddings python /app/generate_embeddings.py \\
  --input {remote_input} \\
  --output {output_dir} \\
  --model {model} \\
  --repr-layers {repr_layers_str} \\
  {mean_flag} \\
  {tok_flag} \\
  --gpu {gpu_id} \\
  2>&1 | tee {remote_job_path}/esm.log

if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi
"""
        return script
    
    # Keep wait(), download(), list_jobs() unchanged...
    
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
        """Download embedding results"""
        status_info = self.status(job_id)
        
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")
        
        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']
        
        # Create local output directory
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download embeddings directory
        import tarfile
        remote_tar = f"{remote_path}/embeddings.tar.gz"
        local_tar = local_dir / f"{job_id}_embeddings.tar.gz"
        
        # Create tar on remote
        self.conn.run(
            f'cd {remote_path} && tar -czf embeddings.tar.gz embeddings/',
            hide=True
        )
        
        # Download
        self.conn.get(remote_tar, str(local_tar))
        
        # Extract locally
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(local_dir)
        
        # Download log
        local_log = local_dir / f"{job_id}_log.txt"
        self.conn.get(f"{remote_path}/esm.log", str(local_log))
        
        print(f"✓ Downloaded embeddings to {local_dir / 'embeddings'}")
        print(f"✓ Downloaded log to {local_log}")
        
        return local_dir / 'embeddings'
    
    def list_jobs(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """List all embedding jobs"""
        job_db = self._load_job_db()
        
        # Filter for embedding jobs
        embed_jobs = {k: v for k, v in job_db.items() if v.get('job_type') == 'embeddings'}
        
        if not embed_jobs:
            return pd.DataFrame()
        
        jobs_list = []
        for job_id, info in embed_jobs.items():
            current_status = self.status(job_id)
            
            if status_filter and current_status['status'] != status_filter:
                continue
            
            jobs_list.append({
                'job_id': job_id,
                'name': info['name'],
                'status': current_status['status'],
                'model': info.get('model', 'unknown'),
                'submitted': info['submitted_at'],
                'runtime': current_status['runtime']
            })
        
        df = pd.DataFrame(jobs_list)
        if not df.empty:
            df = df.sort_values('submitted', ascending=False)
        
        return df