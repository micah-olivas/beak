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

    # Structured fields used by `estimate_peak_vram_gb` to model
    # activation memory. `flash_attn` is what flips the formula from
    # O(L²) (original PyTorch attention) to O(L) (FlashAttention) —
    # fair-esm in the embedding container does *not* enable
    # FlashAttention for ESM-2 [1]; the `esm` package's ESM-C path
    # does. This drives most of the practical OOM differences at
    # long sequences.
    #
    # [1] Çelik & Xie 2025 — "the original implementation cannot
    #     handle proteins longer than 3,000 amino acids for billion
    #     parameter models" without FlashAttention.
    AVAILABLE_MODELS = {
        'esm2_t6_8M_UR50D': {
            'name': 'ESM2-8M',
            'family': 'esm2',
            'params': '8M',
            'vram_gb': 0.1,
            'n_layers': 6, 'hidden_dim': 320, 'n_heads': 20,
            'flash_attn': False,
            'description': 'ESM2, 6 layers, 320 dim',
        },
        'esm2_t12_35M_UR50D': {
            'name': 'ESM2-35M',
            'family': 'esm2',
            'params': '35M',
            'vram_gb': 0.2,
            'n_layers': 12, 'hidden_dim': 480, 'n_heads': 20,
            'flash_attn': False,
            'description': 'ESM2, 12 layers, 480 dim',
        },
        'esm2_t30_150M_UR50D': {
            'name': 'ESM2-150M',
            'family': 'esm2',
            'params': '150M',
            'vram_gb': 0.6,
            'n_layers': 30, 'hidden_dim': 640, 'n_heads': 20,
            'flash_attn': False,
            'description': 'ESM2, 30 layers, 640 dim',
        },
        'esm2_t33_650M_UR50D': {
            'name': 'ESM2-650M',
            'family': 'esm2',
            'params': '650M',
            'vram_gb': 2.5,
            'n_layers': 33, 'hidden_dim': 1280, 'n_heads': 20,
            'flash_attn': False,
            'description': 'ESM2, 33 layers, 1280 dim',
        },

        # ESM-C family (EvolutionaryScale). Default `esm` Python
        # package uses FlashAttention, so memory scales linearly
        # with L instead of quadratically.
        'esmc_300m': {
            'name': 'ESM-C-300M',
            'family': 'esmc',
            'params': '300M',
            'vram_gb': 1.5,
            'n_layers': 30, 'hidden_dim': 960, 'n_heads': 15,
            'flash_attn': True,
            'description': 'ESM-C (2024-12), 30 layers, 960 dim',
        },
        'esmc_600m': {
            'name': 'ESM-C-600M',
            'family': 'esmc',
            'params': '600M',
            'vram_gb': 2.5,
            'n_layers': 36, 'hidden_dim': 1152, 'n_heads': 18,
            'flash_attn': True,
            'description': 'ESM-C (2024-12), 36 layers, 1152 dim',
        },
    }

    @classmethod
    def estimate_peak_vram_gb(
        cls, model_id: str, max_len: int,
        flash_attn_available: bool = True,
    ) -> float:
        """Rough peak VRAM (GB) for embedding one sequence of length `max_len`.

        FlashAttention support is hardware-gated: it needs at least
        Turing (SM 7.5+) and FA-2 needs Ampere (SM 8.0+). Pre-Ampere
        cards (Pascal: 1080 Ti, P100; Volta: V100 in some configs)
        fall back to PyTorch's `math` attention even when the model
        code requests SDPA — so ESM-C's "linear in L" benefit
        evaporates on those GPUs. Pass `flash_attn_available=False`
        to model that case.

        Two regimes:

        - **No FlashAttention** (Pascal hardware OR ESM-2's fair-esm
          path) — activations are dominated by the attention score
          matrix at every layer:
          `~ n_layers × n_heads × L² × 2 bytes` (bf16). The 1.5×
          fudge folds in q/k/v projections, MLP, and layer norms
          without overcomplicating the formula.
        - **FlashAttention** (Ampere+ AND model code that uses it) —
          linear in L, dominated by hidden-state buffers:
          `~ 6 × n_layers × L × hidden_dim × 2 bytes`. The factor 6
          accounts for the residual stream + attention output + MLP
          intermediate; consistent with the per-sequence slice of
          ESME paper's batch-of-16 numbers (3B at L=3500: ~525 MB).

        Numbers should be read as "rough order of magnitude" —
        useful for distinguishing comfortable / tight / OOM, not for
        predicting allocator behavior to the megabyte.
        """
        info = cls.AVAILABLE_MODELS.get(model_id)
        if not info:
            return 0.0
        weights_gb = info.get('vram_gb', 0.0)
        n_layers = info.get('n_layers') or 1
        # Effective FlashAttention = model wants it AND hardware supports it.
        # Without hardware support, even ESM-C falls back to math attention.
        use_flash = info.get('flash_attn') and flash_attn_available
        if use_flash:
            hidden = info.get('hidden_dim') or 1280
            act_bytes = 6 * n_layers * max_len * hidden * 2
        else:
            n_heads = info.get('n_heads') or 20
            act_bytes = 1.5 * n_layers * n_heads * (max_len ** 2) * 2
        return weights_gb + act_bytes / (1024 ** 3)

    # CUDA compute-capability gate for FlashAttention. We use the
    # FA-2 threshold (SM 8.0, Ampere) because PyTorch's stable SDPA
    # flash backend (`scaled_dot_product_attention` with FLASH) has
    # been FA-2-based since torch 2.x. FA-1 (SM 7.5, Turing) is
    # legacy and not reliably installable from PyPI for current
    # torch versions.
    _FLASH_ATTN_MIN_COMPUTE_CAPABILITY = (8, 0)

    @classmethod
    def gpu_supports_flash_attn(
        cls, gpu_name: str = "", compute_cap: str = "",
    ) -> bool:
        """Does this GPU support FlashAttention?

        Prefers the explicit compute-capability check (authoritative)
        over name matching. `compute_cap` comes from
        `nvidia-smi --query-gpu=compute_cap` and looks like "6.1"
        (Pascal) / "7.5" (Turing) / "8.0" (Ampere) / "9.0" (Hopper).
        Falls back to a name-based heuristic for ancient drivers
        that don't expose the field.

        Conservative when ambiguous: returns True (assume modern
        hardware) so users on unrecognised cards don't get spurious
        warnings — an actual OOM at submit time is more actionable
        than a false alarm here.
        """
        if compute_cap:
            try:
                major, minor = compute_cap.split(".")
                cc = (int(major), int(minor))
            except (ValueError, AttributeError):
                cc = None
            if cc is not None:
                return cc >= cls._FLASH_ATTN_MIN_COMPUTE_CAPABILITY

        # Name fallback for legacy drivers without compute_cap. Lists
        # are not exhaustive — when the name doesn't match any known
        # pre-Ampere card, default to True. The known-bad list
        # below covers the most common "lab server with old GPUs"
        # cases (Pascal/Maxwell/Kepler).
        if not gpu_name:
            return True
        n = gpu_name.lower()
        pascal = (
            "1080", "1070", "1060", "1050",
            " p100", " p40", " p4 ",
            "titan x", "titan xp",
        )
        even_older = (
            "980", "970", "960", "950", "940", "930",
            "780", "770", "760", "750", "k80", "k40", "m40",
        )
        # Volta (V100, Titan V) is SM 7.0 — no FA-2 even though it
        # has tensor cores. List separately for clarity.
        volta = ("v100", "titan v")
        for tag in pascal + even_older + volta:
            if tag in n:
                return False
        return True

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

    def query_gpus(self) -> List[Dict]:
        """Return a list of {index, free_mb, total_mb, name, compute_cap}
        for each GPU on the remote.

        `compute_cap` is the CUDA Compute Capability as a string
        (e.g. "6.1" for Pascal, "8.0" for Ampere, "9.0" for Hopper) —
        the canonical signal for whether FlashAttention will run.
        Empty string when `nvidia-smi` doesn't expose the field
        (very old driver) so callers can fall back to name matching.

        Relies on `nvidia-smi` being on the remote PATH. If nvidia-smi
        isn't available (e.g. AMD GPUs, CPU-only host, or no NVIDIA
        driver installed), returns an empty list — callers should
        treat that as "can't tell, skip the check" rather than an
        error. CPU-only deployments are useful for testing the
        pipeline but will be very slow for real workloads.
        """
        # Newer nvidia-smi (>=470) supports compute_cap as a queryable
        # field. We try the rich query first; if that returns nothing
        # (older driver), fall back to the legacy 4-column query so
        # users on ancient drivers still get a memory budget.
        rich = self.conn.run(
            'nvidia-smi --query-gpu=index,memory.free,memory.total,name,'
            'compute_cap --format=csv,noheader,nounits 2>/dev/null',
            hide=True, warn=True,
        )
        if rich.ok and rich.stdout.strip():
            return self._parse_gpu_csv(rich.stdout, with_compute_cap=True)

        legacy = self.conn.run(
            'nvidia-smi --query-gpu=index,memory.free,memory.total,name '
            '--format=csv,noheader,nounits 2>/dev/null',
            hide=True, warn=True,
        )
        if legacy.ok and legacy.stdout.strip():
            return self._parse_gpu_csv(legacy.stdout, with_compute_cap=False)
        return []

    @staticmethod
    def _parse_gpu_csv(text: str, with_compute_cap: bool) -> List[Dict]:
        gpus = []
        min_fields = 5 if with_compute_cap else 4
        for line in text.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < min_fields:
                continue
            try:
                row = {
                    'index': int(parts[0]),
                    'free_mb': int(parts[1]),
                    'total_mb': int(parts[2]),
                    'name': parts[3],
                    'compute_cap': parts[4] if with_compute_cap else '',
                }
            except ValueError:
                continue
            gpus.append(row)
        return gpus

    def _select_gpu(self, requested: int, model: str) -> int:
        """Resolve the GPU to use, honoring `requested` (an int or 'auto').

        For an integer: check that the chosen GPU has at least 2× the
        model's advertised VRAM footprint free, and abort with a helpful
        message if not. For 'auto': pick the GPU with the most free
        memory, printing the choice.
        """
        vram_gb = self.AVAILABLE_MODELS[model]['vram_gb']
        required_mb = int(vram_gb * 1024 * 2)  # 2× headroom for activations

        gpus = self.query_gpus()
        if not gpus:
            # Can't see the GPUs — fall through, let CUDA produce the error.
            return int(requested) if requested != 'auto' else 0

        if requested == 'auto':
            best = max(gpus, key=lambda g: g['free_mb'])
            if best['free_mb'] < required_mb:
                summary = ', '.join(
                    f"gpu{g['index']}: {g['free_mb']} MB free" for g in gpus
                )
                raise RuntimeError(
                    f"No GPU has >= {required_mb} MB free "
                    f"(model {model} needs ~{vram_gb} GB, 2× headroom). "
                    f"Current state: {summary}"
                )
            print(
                f"Auto-selected GPU {best['index']} "
                f"({best['free_mb']} MB free of {best['total_mb']}, {best['name']})"
            )
            return best['index']

        idx = int(requested)
        match = next((g for g in gpus if g['index'] == idx), None)
        if match is None:
            raise RuntimeError(
                f"GPU {idx} not found on remote. Available: "
                f"{[g['index'] for g in gpus]}"
            )
        if match['free_mb'] < required_mb:
            raise RuntimeError(
                f"GPU {idx} has only {match['free_mb']} MB free; "
                f"model {model} needs ~{required_mb} MB (2× headroom for "
                f"activations). Pass --gpu auto to pick the freest GPU, "
                f"or wait for the current job to finish."
            )
        return idx

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
               input_file: Optional[str] = None,
               remote_input: Optional[str] = None,
               source_job_id: Optional[str] = None,
               model: str = 'esm2_t33_650M_UR50D',
               job_name: Optional[str] = None,
               repr_layers: List[int] = [-1],
               include_mean: bool = True,
               include_per_tok: bool = False,
               gpu_id=0) -> str:
        """
        Submit ESM embedding generation job.

        Exactly one of `input_file` or `remote_input` must be provided.

        Args:
            input_file: Path to a local FASTA file; uploaded to the remote
                job directory as input.fasta.
            remote_input: Absolute path to a FASTA file already on the
                remote (e.g. a previous job's hits.fasta). Copied — not
                moved — into the new job's directory, so the source job
                stays intact.
            source_job_id: When `remote_input` came from another beak
                job, pass its id here so the new job's record tracks
                the lineage (jobs.json[<new_id>]['source_job_id']).
            model: ESM model ID (use list_models() to see options)
            job_name: Optional human-readable job name
            repr_layers: Which layers to extract ([-1] = last layer)
            include_mean: Include mean pooled embeddings
            include_per_tok: Include per-token embeddings (large files!)
            gpu_id: GPU device ID (int) or the string 'auto' to pick
                the GPU with the most free memory. Before launch we
                check that the selected GPU has at least 2× the model's
                advertised VRAM footprint free; otherwise we abort with
                a helpful message rather than silently OOMing.

        Returns:
            job_id: Unique job identifier
        """
        if (input_file is None) == (remote_input is None):
            raise ValueError(
                "Pass exactly one of `input_file` (local) or `remote_input` "
                "(already-on-remote path)."
            )

        if model not in self.AVAILABLE_MODELS:
            available = ', '.join(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Available: {available}")

        # Resolve gpu_id — accepts int or 'auto'. Raises with an
        # actionable message if the chosen GPU is too busy.
        gpu_id = self._select_gpu(gpu_id, model)

        job_id = str(uuid.uuid4())[:8]
        if not job_name:
            from .naming import generate_readable_name
            job_name = f"esm_{generate_readable_name()}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)
        # Embeddings are written under {remote_job_path}/embeddings/ by
        # generate_embeddings.py. Pre-create it so we can stage the
        # inherited taxonomy TSV alongside the pickles — it'll ride back
        # in the results tarball automatically.
        embeddings_dir = f"{remote_job_path}/embeddings"
        self.conn.run(f'mkdir -p {embeddings_dir}', hide=True)

        job_remote_input = f"{remote_job_path}/input.fasta"
        if input_file is not None:
            self.conn.put(input_file, job_remote_input)
            input_record = str(input_file)
        else:
            # Copy rather than move: source job dir stays intact.
            check = self.conn.run(
                f'[ -f {remote_input} ] && echo OK || echo MISSING',
                hide=True, warn=True,
            )
            if check.stdout.strip() != 'OK':
                raise FileNotFoundError(
                    f"Remote FASTA not found on {self.conn.host}: {remote_input}"
                )
            self.conn.run(f'cp {remote_input} {job_remote_input}', hide=True)
            input_record = f"remote:{remote_input}"

            # Carry hit taxonomy forward from the source search job so
            # downstream plotting (e.g. PCA colored by domain) has it
            # locally, without a second network round-trip from the user.
            if source_job_id:
                self._inherit_hit_taxonomy(source_job_id, embeddings_dir)
        remote_input = job_remote_input  # used by the job script below

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

        with self._mutate_job_db() as db:
            db[job_id] = {
                'job_type': 'embeddings',
                'name': job_name,
                'model': model,
                'input_file': input_record,
                'source_job_id': source_job_id,
                'remote_path': remote_job_path,
                'submitted_at': datetime.now().isoformat(),
                'status': 'SUBMITTED',
                'pid': pid,
                'parameters': {
                    'repr_layers': repr_layers,
                    'include_mean': include_mean,
                    'include_per_tok': include_per_tok,
                    'gpu_id': gpu_id,
                },
            }

        model_info = self.AVAILABLE_MODELS[model]
        print(f"✓ Submitted {job_name} → {model_info['name']} ({job_id})")

        return job_id

    def _inherit_hit_taxonomy(self, source_job_id: str,
                              dest_embeddings_dir: str) -> None:
        """Copy the source job's hits_taxonomy.tsv into this embeddings dir.

        Best-effort: if the source is a search whose target DB was a
        seqTaxDB, we attempt (and cache) the taxonomy extraction via
        :meth:`MMseqsSearch.ensure_remote_hits_taxonomy`. Silent no-op
        otherwise — the caller hasn't asked for it, we just set them
        up for success if it's available.
        """
        job_db = self._load_job_db()
        src = job_db.get(source_job_id)
        if not src or src.get('job_type') != 'search':
            return  # Only search sources carry hit taxonomy today.

        try:
            from .search import MMseqsSearch
            src_mgr = MMseqsSearch(connection=self.conn)
            remote_tsv = src_mgr.ensure_remote_hits_taxonomy(source_job_id)
        except Exception as exc:  # noqa: BLE001 — taxonomy is opportunistic
            print(f"  (couldn't inherit taxonomy from {source_job_id}: {exc})")
            return

        if not remote_tsv:
            return

        self.conn.run(
            f'cp {remote_tsv} {dest_embeddings_dir}/hits_taxonomy.tsv',
            hide=True, warn=True,
        )
        print(f"  Inherited hit taxonomy from source job {source_job_id}")

    def _generate_job_script(self, remote_job_path, remote_input, model,
                        repr_layers, include_mean, include_per_tok, gpu_id):
        """Generate Docker execution script"""
        docker_dir, project_name, _ = self._resolve_docker_dir()
        output_dir = f"{remote_job_path}/embeddings"

        repr_layers_str = ' '.join(map(str, repr_layers))
        mean_flag = "--include-mean" if include_mean else ""
        tok_flag = "--include-per-tok" if include_per_tok else ""

        # ESM-C uses EvolutionaryScale's `esm` SDK, which lives in a
        # separate venv inside the container so it doesn't collide
        # with fair-esm's `esm` import. Dispatch the right Python.
        if model.startswith("esmc_") or "/esmc" in model.lower():
            container_python = "/opt/esmc-venv/bin/python"
        else:
            container_python = "python"

        # set -o pipefail is critical: without it, `exec | tee` swallows the
        # exec exit code (tee is always 0) and the script wrongly writes
        # COMPLETED when the container exec actually failed.
        script = f"""#!/bin/bash
set -eo pipefail

echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

mkdir -p {output_dir}

cd {docker_dir}

# Wait for the embeddings container to be ready for exec (avoids the
# "procReady not received" race right after `docker compose up -d`).
READY=false
for attempt in $(seq 1 30); do
    if docker compose --project-name {project_name} exec -T embeddings true 2>/dev/null; then
        READY=true
        break
    fi
    sleep 2
done
if [ "$READY" != "true" ]; then
    echo "Container 'embeddings' was not ready for exec after 60s" \\
        | tee -a {remote_job_path}/esm.log
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
    exit 1
fi

set +e
docker compose --project-name {project_name} exec -T embeddings {container_python} /app/generate_embeddings.py \\
  --input {remote_input} \\
  --output {output_dir} \\
  --model {model} \\
  --repr-layers {repr_layers_str} \\
  {mean_flag} \\
  {tok_flag} \\
  --gpu {gpu_id} \\
  2>&1 | tee {remote_job_path}/esm.log
EXIT_CODE=${{PIPESTATUS[0]}}
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date) (exit $EXIT_CODE)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
    exit $EXIT_CODE
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

    def detailed_status(self, job_id: str) -> Dict:
        """Base detailed status, augmented with ESM progress counters.

        When the job has written a progress.json, include a
        `embedding_progress` sub-dict with {done, total, failed, current}
        so the watch display can show "42/1000 — current: P14621" in real
        time without the user having to tail the remote log.
        """
        info = super().detailed_status(job_id)
        if info.get('status') in ('RUNNING', 'SUBMITTED'):
            try:
                p = self.progress(job_id)
            except Exception:
                p = {'status': 'NO_PROGRESS'}
            if p.get('status') != 'NO_PROGRESS':
                info['embedding_progress'] = {
                    'done': p.get('done', 0),
                    'total': p.get('total', 0),
                    'failed': p.get('failed', 0),
                    'current': p.get('current'),
                    'model': p.get('model'),
                    'last_error': p.get('last_error'),
                }
        return info

    def progress(self, job_id: str) -> Dict:
        """Fetch structured progress for a running embedding job.

        The in-container generator writes a progress.json beside the
        embeddings output directory; this reads it back over SSH without
        downloading the full result tarball.

        Returns a dict with keys: total, done, failed, current (seq_id or
        None), started_at, last_update, model. Returns {'status': 'NO_PROGRESS'}
        if the job hasn't written the file yet (e.g., still staging).
        """
        import json

        job_db = self._load_job_db()
        if job_id not in job_db:
            raise ValueError(f"Unknown job id: {job_id}")

        remote_path = job_db[job_id]['remote_path']
        progress_path = f"{remote_path}/embeddings/progress.json"

        result = self.conn.run(
            f'cat {progress_path} 2>/dev/null || echo __MISSING__',
            hide=True, warn=True,
        )
        text = result.stdout.strip()
        if text == '__MISSING__' or not text:
            return {'status': 'NO_PROGRESS'}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Partial write caught mid-flush; treat as transient
            return {'status': 'NO_PROGRESS'}

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
