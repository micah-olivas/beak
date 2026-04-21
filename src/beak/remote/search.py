import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union

from .base import RemoteJobManager


class MMseqsSearch(RemoteJobManager):
    """MMseqs2 search job manager for remote execution"""

    JOB_TYPE = 'search'
    LOG_FILE = 'mmseqs.log'
    LOG_OPERATIONS = [
        ('Index table: counting k-mers', 'Counting k-mers'),
        ('Index table: fill', 'Building index'),
        ('Starting prefiltering scores', 'Computing scores'),
        ('align', 'Aligning'),
        ('convertalis', 'Finalizing'),
    ]

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

    PRESETS = {
        'default': {
            'description': 'Default MMseqs2 sensitivity (balanced speed/sensitivity)',
            'params': {
                's': 5.7,
                'e': 0.001,
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
                's': 8.5,
                'e': 10,
                'max_seqs': 30000,
                'min_seq_id': 0.15,
                'c': 0.3,
                'cov_mode': 0,
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

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        return {'database': info.get('database', '')}

    def submit(self,
           query_file: str,
           database: str,
           job_name: Optional[str] = None,
           preset: Optional[str] = None,
           **mmseqs_params) -> str:
        """Submit a new MMseqs2 search job"""
        job_id = str(uuid.uuid4())[:8]

        project_dir = self.create_project(
            job_id=job_id,
            job_type='search',
            name=job_name,
            query_file=query_file
        )

        if not job_name:
            from .naming import generate_readable_name
            job_name = f"search_{generate_readable_name()}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        # Apply preset if specified
        final_params = {}
        if preset:
            if preset not in self.PRESETS:
                available = ', '.join(self.PRESETS.keys())
                raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

            final_params = self.PRESETS[preset]['params'].copy()

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

        # Upload query file
        remote_query = f"{remote_job_path}/query.fasta"
        self.conn.put(query_file, remote_query)

        # Format MMseqs2 parameters
        param_str = []
        for k, v in final_params.items():
            param_name = k.replace("_", "-")
            prefix = "-" if len(k) == 1 else "--"
            param_str.append(f'{prefix}{param_name} {v}')
        param_str = ' '.join(param_str)

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

# Convert to m8 format
mmseqs convertalis \\
  {remote_job_path}/queryDB \\
  {db_path} \\
  {remote_job_path}/resultDB \\
  {remote_job_path}/results.m8 \\
  2>&1 | tee -a {remote_job_path}/mmseqs.log

if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi

rm -rf {remote_job_path}/tmp
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

        print(f"✓ Submitted {job_name} → {database} ({job_id})")

        return job_id

    def get_results(self,
                job_id: str,
                parse: bool = True,
                download_sequences: bool = False) -> Union[Path, pd.DataFrame, Dict]:
        """
        Download and optionally parse job results

        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame; if False, return Path to m8 file
            download_sequences: Also download hit sequences as FASTA

        Returns:
            DataFrame (if parse=True), Path (if parse=False), or Dict (if download_sequences=True)
        """
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")

        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            project_dir = self.create_project(job_id, 'search')

        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']

        # Download m8 results
        m8_file = project_dir / "results.m8"
        if not m8_file.exists():
            self.conn.get(f"{remote_path}/results.m8", str(m8_file))

        # Download log
        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(f"{remote_path}/mmseqs.log", str(log_file))

        # Download sequences if requested
        if download_sequences:
            fasta_file = project_dir / "hits.fasta"
            if not fasta_file.exists():
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
                print(f"✓ Downloaded {n_seqs} hit sequences")

        if parse:
            columns = [
                'query', 'target', 'identity', 'alignment_length',
                'mismatches', 'gap_openings', 'q_start', 'q_end',
                't_start', 't_end', 'evalue', 'bit_score'
            ]
            df = pd.read_csv(m8_file, sep='\t', names=columns, comment='#')

            if download_sequences:
                return {'dataframe': df, 'm8': m8_file, 'fasta': fasta_file}
            return df

        if download_sequences:
            return {'m8': m8_file, 'fasta': fasta_file}

        return m8_file

