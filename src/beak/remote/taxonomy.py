import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from .base import RemoteJobManager


class MMseqsTaxonomy(RemoteJobManager):
    """MMseqs2 taxonomy assignment for remote execution"""

    JOB_TYPE = 'taxonomy'
    LOG_FILE = 'mmseqs.log'
    LOG_OPERATIONS = [
        ('Index table: counting k-mers', 'Counting k-mers'),
        ('Index table: fill', 'Building index'),
        ('Starting prefiltering scores', 'Computing scores'),
        ('align', 'Aligning'),
        ('lca', 'Computing LCA'),
        ('createtsv', 'Generating results'),
    ]

    AVAILABLE_DBS = {
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

        for alias, db_name in self.AVAILABLE_DBS.items():
            db_path = f"{self.DB_BASE_PATH}/{db_name}"

            result = self.conn.run(
                f'if [ -e {db_path} ]; then du -sh {db_path} 2>/dev/null | cut -f1; else echo "NOT_FOUND"; fi',
                hide=True, warn=True
            )

            size = result.stdout.strip()
            exists = size != "NOT_FOUND"

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

    def _list_jobs_extra_columns(self, info: Dict) -> Dict:
        return {'database': info.get('database', '')}

    def submit(self,
           query_file: str,
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
            tax_lineage: If True, include full taxonomic lineage
            **mmseqs_params: Additional MMseqs2 taxonomy parameters

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:8]
        if not job_name:
            from .naming import generate_readable_name
            job_name = f"taxonomy_{generate_readable_name()}"

        project_dir = self.create_project(
            job_id=job_id,
            job_type='taxonomy',
            name=job_name,
            query_file=query_file
        )
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

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

        # Upload query file
        remote_query = f"{remote_job_path}/query.fasta"
        self.conn.put(query_file, remote_query)

        # Format parameters
        param_str = []
        if tax_lineage:
            param_str.append('--tax-lineage 1')

        for k, v in mmseqs_params.items():
            param_name = k.replace("_", "-")
            prefix = "-" if len(k) == 1 else "--"
            param_str.append(f'{prefix}{param_name} {v}')

        param_str = ' '.join(param_str)

        job_script = f"""#!/bin/bash
set -e

echo "Job started: $(date)" > {remote_job_path}/status.txt
echo "RUNNING" >> {remote_job_path}/status.txt

mmseqs createdb {remote_query} {remote_job_path}/queryDB \\
  2>&1 | tee {remote_job_path}/mmseqs.log

mmseqs taxonomy \\
  {remote_job_path}/queryDB \\
  {db_path} \\
  {remote_job_path}/taxonomyResult \\
  {remote_job_path}/tmp \\
  {param_str} \\
  2>&1 | tee -a {remote_job_path}/mmseqs.log

mmseqs createtsv \\
  {remote_job_path}/queryDB \\
  {remote_job_path}/taxonomyResult \\
  {remote_job_path}/results.tsv \\
  2>&1 | tee -a {remote_job_path}/mmseqs.log

if [ $? -eq 0 ]; then
    echo "Job completed: $(date)" >> {remote_job_path}/status.txt
    echo "COMPLETED" >> {remote_job_path}/status.txt
else
    echo "Job failed: $(date)" >> {remote_job_path}/status.txt
    echo "FAILED" >> {remote_job_path}/status.txt
fi

rm -rf {remote_job_path}/tmp {remote_job_path}/queryDB* {remote_job_path}/taxonomyResult*
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

        print(f"✓ Submitted {job_name} → {database} ({job_id})")

        return job_id

    def annotate_search_results(self,
                           hits_fasta: Path,
                           database: str = 'uniprotkb',
                           job_name: Optional[str] = None,
                           tax_lineage: bool = True,
                           **mmseqs_params) -> str:
        """Convenience wrapper to annotate taxonomy on search result FASTA"""
        return self.submit(
            query_file=str(hits_fasta),
            database=database,
            job_name=job_name,
            tax_lineage=tax_lineage,
            **mmseqs_params
        )

    def parse_taxonomy_lineage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the taxonomic lineage string into separate columns"""
        if 'lineage' not in df.columns:
            print("No lineage column found. Run with --tax-lineage 1")
            return df

        def parse_lineage(lineage_str):
            if pd.isna(lineage_str) or lineage_str == '':
                return {}

            ranks = {
                '-': 'unranked',
                'd': 'superkingdom',
                'k': 'domain',
                'p': 'phylum',
                'c': 'class',
                'o': 'order',
                'f': 'family',
                'g': 'genus',
                's': 'species'
            }

            taxonomy = {}
            parts = lineage_str.split(';')

            full_lineage = lineage_str.lower()
            if 'bacteria' in full_lineage and 'archaea' not in full_lineage:
                taxonomy['domain'] = 'Bacteria'
            elif 'archaea' in full_lineage:
                taxonomy['domain'] = 'Archaea'
            elif 'eukaryota' in full_lineage:
                taxonomy['domain'] = 'Eukaryota'

            for part in parts:
                part = part.strip()
                if '_' in part:
                    rank_code, name = part.split('_', 1)

                    if not name or name == '':
                        continue

                    if rank_code == 'k':
                        if 'domain' not in taxonomy:
                            taxonomy['domain'] = name
                        taxonomy['kingdom'] = name
                    elif rank_code in ranks:
                        rank_name = ranks[rank_code]
                        if rank_name != 'unranked':
                            taxonomy[rank_name] = name

            return taxonomy

        lineage_data = df['lineage'].apply(parse_lineage)

        rank_columns = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

        for rank in rank_columns:
            df[rank] = lineage_data.apply(lambda x: x.get(rank, None))

        return df

    def get_results(self, job_id: str, parse: bool = True, parse_lineage: bool = True):
        """
        Download and optionally parse taxonomy results

        Args:
            job_id: Job identifier
            parse: If True, return parsed DataFrame
            parse_lineage: If True and parse=True, parse lineage into separate columns

        Returns:
            Path or DataFrame depending on parse parameter
        """
        status_info = self.status(job_id)
        if status_info['status'] != 'COMPLETED':
            raise ValueError(f"Job not completed (status: {status_info['status']})")

        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            project_dir = self.create_project(job_id, 'taxonomy')

        job_db = self._load_job_db()
        remote_path = job_db[job_id]['remote_path']

        results_file = project_dir / "taxonomy_results.tsv"
        if not results_file.exists():
            self.conn.get(
                remote=f"{remote_path}/results.tsv",
                local=str(results_file)
            )

        log_file = project_dir / "job.log"
        if not log_file.exists():
            self.conn.get(
                remote=f"{remote_path}/mmseqs.log",
                local=str(log_file)
            )

        if not parse:
            return results_file

        df = pd.read_csv(results_file, sep='\t', header=None)

        if len(df.columns) == 5:
            df.columns = ['query', 'taxid', 'rank', 'scientific_name', 'lineage']
        elif len(df.columns) == 4:
            df.columns = ['query', 'taxid', 'rank', 'scientific_name']
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]

        if parse_lineage and 'lineage' in df.columns:
            df = self.parse_taxonomy_lineage(df)

        return df
