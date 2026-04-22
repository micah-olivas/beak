import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from .base import RemoteJobManager


class ESMEmbeddings(RemoteJobManager):
    """ESM protein language model embeddings via Docker"""

    JOB_TYPE = 'embeddings'
    LOG_FILE = 'esm.log'

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

    def __init__(self, host: Optional[str] = None, user: Optional[str] = None,
                 key_path: Optional[str] = None,
                 remote_job_dir: Optional[str] = None, connection=None):
        super().__init__(host, user, key_path, remote_job_dir, connection)
        self._verify_docker()
        self._ensure_docker_service()

    def _verify_docker(self):
        """Verify Docker is available on remote"""
        result = self.conn.run('docker --version', hide=True, warn=True)
        if not result.ok:
            raise RuntimeError(
                "Docker not found on remote server. "
                "Please install Docker or use a different execution mode."
            )

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        return {'model': info.get('model', 'unknown')}

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
        if not job_name:
            from .naming import generate_readable_name
            job_name = f"esm_{generate_readable_name()}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)

        remote_input = f"{remote_job_path}/input.fasta"
        self.conn.put(input_file, remote_input)

        job_script = self._generate_job_script(
            remote_job_path, remote_input, model, repr_layers,
            include_mean, include_per_tok, gpu_id
        )

        script_path = f"{remote_job_path}/run.sh"
        self.conn.put(
            local=self._write_temp_script(job_script),
            remote=script_path
        )
        self.conn.run(f'chmod +x {script_path}', hide=True)

        result = self.conn.run(
            f'nohup {script_path} > {remote_job_path}/nohup.out 2>&1 & echo $!',
            hide=True
        )
        pid = result.stdout.strip()

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
        print(f"✓ Submitted {job_name} → {model_info['name']} ({job_id})")

        return job_id

    def _generate_job_script(self, remote_job_path, remote_input, model,
                        repr_layers, include_mean, include_per_tok, gpu_id):
        """Generate Docker execution script"""
        docker_dir = f"{self.remote_job_dir}/docker"
        output_dir = f"{remote_job_path}/embeddings"

        repr_layers_str = ' '.join(map(str, repr_layers))
        mean_flag = "--include-mean" if include_mean else ""
        tok_flag = "--include-per-tok" if include_per_tok else ""

        script = f"""#!/bin/bash
set -e

echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

mkdir -p {output_dir}

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

    def download(self, job_id: str, local_dir: str = '.') -> Path:
        """Download embedding results"""
        status_info = self.status(job_id)

        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")

        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        import tarfile
        remote_tar = f"{remote_path}/embeddings.tar.gz"
        local_tar = local_dir / f"{job_id}_embeddings.tar.gz"

        self.conn.run(
            f'cd {remote_path} && tar -czf embeddings.tar.gz embeddings/',
            hide=True
        )

        self.conn.get(remote_tar, str(local_tar))

        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(local_dir)

        local_log = local_dir / f"{job_id}_log.txt"
        self.conn.get(f"{remote_path}/esm.log", str(local_log))

        print(f"✓ Downloaded embeddings to {local_dir / 'embeddings'}")
        print(f"✓ Downloaded log to {local_log}")

        return local_dir / 'embeddings'

    def get_results(self, job_id: str, parse: bool = True,
                    kind: str = 'mean', layer=None, local_dir: str = '.'):
        """Retrieve embedding results.

        Args:
            job_id: job to fetch.
            parse: if True (default), return a pandas DataFrame via the
                beak.embeddings loader; if False, return the Path to the
                downloaded embeddings/ directory.
            kind: 'mean' or 'per_token' — which pickle to load.
            layer: layer selector passed through to the loader.
            local_dir: where to unpack the results tarball.

        Returns:
            DataFrame (parse=True) or Path to embeddings/ (parse=False).
        """
        embeddings_dir = self.download(job_id, local_dir=local_dir)

        if not parse:
            return embeddings_dir

        if kind == 'mean':
            from ..embeddings import load_mean_embeddings
            return load_mean_embeddings(
                embeddings_dir / 'mean_embeddings.pkl', layer=layer
            )
        if kind == 'per_token':
            from ..embeddings import load_per_token_embeddings
            return load_per_token_embeddings(
                embeddings_dir / 'per_token_embeddings.pkl', layer=layer
            )
        raise ValueError(f"Unknown kind '{kind}'. Use 'mean' or 'per_token'.")
