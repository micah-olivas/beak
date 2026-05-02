import re
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from .base import RemoteJobManager


# Algorithm definitions: command template, log file, supported output
# formats, and the keyword→stage map the job-status modal walks to render
# stage progress. Keywords are matched against log lines from newest to
# oldest, so the most recent matching stage is shown as "active".
ALGORITHMS = {
    'clustalo': {
        'name': 'Clustal Omega',
        'log_file': 'clustalo.log',
        'output_formats': ['fasta', 'clustal', 'msf', 'phylip', 'selex', 'stockholm', 'vienna'],
        'default_format': 'fasta',
        'log_operations': [
            ('Read ', 'Reading sequences'),
            ('Calculating pairwise', 'Pairwise distances'),
            ('Guide-tree', 'Building guide tree'),
            ('Progressive alignment', 'Progressive alignment'),
        ],
    },
    'mafft': {
        'name': 'MAFFT',
        'log_file': 'mafft.log',
        'output_formats': ['fasta'],
        'default_format': 'fasta',
        'log_operations': [
            ('Making a distance matrix', 'Building distance matrix'),
            ('Constructing a UPGMA tree', 'Building guide tree'),
            ('Progressive alignment', 'Progressive alignment'),
            ('STEP', 'Progressive alignment'),
            ('Iterative refinement', 'Iterative refinement'),
            ('Iteration', 'Iterative refinement'),
        ],
    },
    'muscle': {
        'name': 'MUSCLE',
        'log_file': 'muscle.log',
        'output_formats': ['fasta'],
        'default_format': 'fasta',
        'log_operations': [
            ('Input:', 'Reading sequences'),
            ('HMM', 'Computing HMMs'),
            ('UPGMA', 'Building UPGMA tree'),
            ('Refining', 'Refining alignment'),
            ('iter', 'Iterative refinement'),
        ],
    },
}


def _build_clustalo_cmd(input_path, output_path, output_format, log_path, params):
    """Build Clustal Omega command string."""
    parts = [f'clustalo -i {input_path} -o {output_path}']
    if output_format != 'fasta':
        parts.append(f'--outfmt={output_format}')
    for k, v in params.items():
        parts.append(f'--{k.replace("_", "-")}={v}')
    parts.append(f'2>&1 | tee {log_path}')
    return ' \\\n  '.join(parts)


def _build_mafft_cmd(input_path, output_path, output_format, log_path, params):
    """Build MAFFT command string."""
    parts = ['mafft']
    # MAFFT uses --flag or --key value style
    for k, v in params.items():
        flag = f'--{k.replace("_", "-")}'
        if isinstance(v, bool) and v:
            parts.append(flag)
        else:
            parts.append(f'{flag} {v}')
    parts.append(input_path)
    # MAFFT writes to stdout
    return ' '.join(parts) + f' > {output_path} 2> {log_path}'


def _build_muscle_cmd(input_path, output_path, output_format, log_path, params):
    """Build MUSCLE command string."""
    parts = [f'muscle -align {input_path} -output {output_path}']
    for k, v in params.items():
        parts.append(f'-{k.replace("_", "-")} {v}')
    parts.append(f'2>&1 | tee {log_path}')
    return ' \\\n  '.join(parts)


_CMD_BUILDERS = {
    'clustalo': _build_clustalo_cmd,
    'mafft': _build_mafft_cmd,
    'muscle': _build_muscle_cmd,
}


class Align(RemoteJobManager):
    """Multiple sequence alignment manager supporting multiple algorithms."""

    JOB_TYPE = 'align'
    LOG_FILE = 'align.log'  # fallback only — actual file resolved per-job

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        return {
            'algorithm': info.get('algorithm', 'clustalo'),
            'output_format': info.get('output_format', 'fasta'),
        }

    def _resolve_log_file(self, job_info: Dict) -> str:
        """Each algorithm writes to its own log file (clustalo.log /
        mafft.log / muscle.log). Without this, `detailed_status` tries
        to tail `align.log`, finds nothing, and the job-status modal
        shows only the bare status header."""
        algo = job_info.get('algorithm', 'clustalo')
        return ALGORITHMS.get(algo, ALGORITHMS['clustalo'])['log_file']

    def _resolve_log_operations(self, job_info: Dict) -> list:
        """Algorithm-specific stage list — used by the job-status modal
        to render the "Stages" block."""
        algo = job_info.get('algorithm', 'clustalo')
        return ALGORITHMS.get(algo, ALGORITHMS['clustalo']).get('log_operations', [])

    def _parse_log_progress(self, log_content: str,
                            log_operations: Optional[list] = None) -> Dict:
        """Extend the base parse with align-specific counters.

        - Sequence count from "Read N sequences" (clustalo) / "Input: N"
          (muscle) / explicit n_homologs is shown so users know how big
          a problem this run is solving.
        - MAFFT prints `STEP X / Y` during progressive alignment; we
          surface that as the align step counter so the modal renders
          a live progress bar.
        """
        progress = super()._parse_log_progress(log_content, log_operations)

        if not log_content or log_content == "No log file":
            return progress

        lines = log_content.strip().split('\n')

        # MAFFT progressive-alignment step counter ("STEP   12 / 38").
        # Walk newest→oldest so the bar reflects current state.
        step_re = re.compile(r'STEP\s+(\d+)\s*/\s*(\d+)')
        for line in reversed(lines):
            m = step_re.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if tot > 0 and cur <= tot:
                    progress['align_step'] = cur
                    progress['total_align_steps'] = tot
                    progress['current_step'] = 'align'
                    break

        # Sequence count — appears once near the top of the log.
        seq_re = re.compile(
            r'(?:Read|Input:)\s+(\d+)\s+(?:sequences|seqs)', re.IGNORECASE
        )
        for line in lines[:60]:
            m = seq_re.search(line)
            if m:
                progress['n_sequences'] = int(m.group(1))
                break

        return progress

    def submit(self,
               input_file: str,
               job_name: Optional[str] = None,
               algorithm: str = 'clustalo',
               output_format: Optional[str] = None,
               **align_params) -> str:
        """
        Submit a multiple sequence alignment job.

        Args:
            input_file: Path to local FASTA file with sequences to align
            job_name: Optional human-readable job name
            algorithm: Alignment algorithm — 'clustalo', 'mafft', or 'muscle'
            output_format: Output format (default depends on algorithm)
            **align_params: Algorithm-specific parameters passed through

        Returns:
            job_id: Unique job identifier
        """
        if algorithm not in ALGORITHMS:
            available = ', '.join(ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

        algo_info = ALGORITHMS[algorithm]
        output_format = output_format or algo_info['default_format']

        if output_format not in algo_info['output_formats']:
            raise ValueError(
                f"Format '{output_format}' not supported by {algo_info['name']}. "
                f"Supported: {', '.join(algo_info['output_formats'])}"
            )

        job_id = str(uuid.uuid4())[:8]
        if not job_name:
            from .naming import generate_readable_name
            job_name = f"align_{generate_readable_name()}"

        self.create_project(
            job_id=job_id,
            job_type='align',
            name=job_name,
            query_file=input_file
        )
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)

        remote_input = f"{remote_job_path}/input.fasta"
        self.conn.put(input_file, remote_input)

        log_path = f"{remote_job_path}/{algo_info['log_file']}"
        output_path = f"{remote_job_path}/alignment.{output_format}"
        filtered_input = f"{remote_job_path}/filtered_input.fasta"

        align_cmd = _CMD_BUILDERS[algorithm](
            filtered_input, output_path, output_format, log_path, align_params
        )

        job_script = f"""#!/bin/bash
set -e

echo "Job started: $(date)" > {remote_job_path}/status.txt
echo 'RUNNING' >> {remote_job_path}/status.txt

# Filter empty sequences
awk '/^>/ {{if (seq) print header "\\n" seq; header=$0; seq=""; next}} {{seq=seq $0}} END {{if (seq) print header "\\n" seq}}' {remote_input} | \\
  awk 'BEGIN {{RS=">"; ORS=""}} NR>1 {{split($0,a,"\\n"); seq=""; for(i=2;i<=length(a);i++) seq=seq a[i]; if (length(seq)>0) print ">" $0}}' > {filtered_input}

SEQ_COUNT=$(grep -c '^>' {filtered_input} || echo 0)

if [ "$SEQ_COUNT" -eq 0 ]; then
    echo "Error: No valid sequences found" | tee -a {log_path}
    echo "FAILED" >> {remote_job_path}/status.txt
    exit 1
fi

# Run {algo_info['name']}
{align_cmd}

if [ $? -eq 0 ]; then
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "FAILED" >> {remote_job_path}/status.txt
fi
"""

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

        self.conn.run(f'echo {pid} > {remote_job_path}/pid.txt', hide=True)

        job_db = self._load_job_db()
        job_db[job_id] = {
            'job_type': 'align',
            'name': job_name,
            'input_file': str(input_file),
            'algorithm': algorithm,
            'output_format': output_format,
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid,
            'parameters': align_params
        }
        self._save_job_db(job_db)

        print(f"✓ Submitted {job_name} [{algo_info['name']}] ({job_id})")

        return job_id

    def get_results(self, job_id: str, parse: bool = False):
        """
        Download and optionally parse alignment results.

        Args:
            job_id: Job identifier
            parse: If True, return BioPython alignment object

        Returns:
            Path to alignment file, or BioPython MultipleSeqAlignment if parse=True
        """
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")

        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            project_dir = self.create_project(job_id, 'align')

        job_db = self._load_job_db()
        job_info = job_db[job_id]
        remote_path = job_info['remote_path']
        output_format = job_info['output_format']
        algorithm = job_info.get('algorithm', 'clustalo')
        log_file_name = ALGORITHMS.get(algorithm, {}).get('log_file', 'align.log')

        alignment_file = project_dir / f"alignment.{output_format}"
        if not alignment_file.exists():
            self.conn.get(
                remote=f"{remote_path}/alignment.{output_format}",
                local=str(alignment_file)
            )

        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(
                remote=f"{remote_path}/{log_file_name}",
                local=str(log_file)
            )

        if not parse:
            return alignment_file

        if output_format != 'fasta':
            print(f"Warning: Parsing only supported for FASTA format, got {output_format}")
            return alignment_file

        from Bio import AlignIO

        alignment = AlignIO.read(alignment_file, "fasta")

        return alignment


# Backwards compatibility
ClustalAlign = Align
