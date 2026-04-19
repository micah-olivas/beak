"""Synchronous hmmscan via SSH for Pfam domain annotation.

Unlike other remote modules, HmmerScan does NOT inherit RemoteJobManager.
A single-sequence hmmscan completes in seconds and does not need background
job tracking, PID management, or project scaffolding.
"""

import uuid
from typing import Dict, List, Optional

from fabric import Connection


PFAM_WELL_KNOWN_PATHS = [
    "/srv/protein_sequence_databases/pfam",
    "~/beak_databases/pfam",
]

PFAM_HMM_FILE = "Pfam-A.hmm"


def resolve_pfam_path(conn: Connection) -> str:
    """Find the Pfam-A HMM database on the remote server.

    Resolution order:
        1. databases.pfam_path from ~/.beak/config.toml
        2. /srv/protein_sequence_databases/pfam/  (system-wide)
        3. ~/beak_databases/pfam/  (user-space)

    Validates that both Pfam-A.hmm and its pressed index (.h3i) exist.

    Args:
        conn: Fabric SSH connection

    Returns:
        Remote path to the directory containing Pfam-A.hmm

    Raises:
        FileNotFoundError: if no valid Pfam database found
    """
    from ..config import get_database_config

    candidates = []

    # Priority 1: explicit config
    db_config = get_database_config()
    if db_config.get('pfam_path'):
        candidates.append(db_config['pfam_path'])

    # Priority 2-3: well-known paths
    candidates.extend(PFAM_WELL_KNOWN_PATHS)

    for path in candidates:
        # Expand tilde on the remote
        if path.startswith('~'):
            home_result = conn.run('echo $HOME', hide=True, warn=True)
            if not home_result.ok:
                continue
            path = home_result.stdout.strip() + path[1:]

        check = conn.run(
            f'[ -f {path}/{PFAM_HMM_FILE} ] && [ -f {path}/{PFAM_HMM_FILE}.h3i ] '
            f'&& echo FOUND || echo MISSING',
            hide=True, warn=True,
        )
        if check.ok and check.stdout.strip() == 'FOUND':
            return path

    raise FileNotFoundError(
        "Pfam database not found on the remote server. "
        "Run 'beak setup pfam' to install, or set the path with: "
        "beak config set databases.pfam_path /path/to/pfam"
    )


class HmmerScan:
    """Run hmmscan against Pfam-A on a remote server via synchronous SSH.

    Usage::

        from beak.remote import BeakSession

        bk = BeakSession()
        hits = bk.hmmer.scan("my_sequence.fasta")
        for hit in hits:
            print(hit['pfam_id'], hit['pfam_name'], hit['i_evalue'])
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    def scan(
        self,
        fasta_path: str,
        pfam_path: Optional[str] = None,
        evalue: float = 1e-5,
    ) -> List[Dict]:
        """Run hmmscan and return parsed Pfam domain hits.

        Args:
            fasta_path: Local path to a FASTA file (single or few sequences)
            pfam_path: Remote directory containing Pfam-A.hmm.
                       Auto-resolved if None.
            evalue: E-value threshold for reporting domains

        Returns:
            List of dicts sorted by i_evalue, each with keys:
                pfam_id, pfam_name, description, evalue, score, bias,
                c_evalue, i_evalue, hmm_from, hmm_to, ali_from, ali_to,
                env_from, env_to
        """
        if pfam_path is None:
            pfam_path = resolve_pfam_path(self.conn)

        tag = str(uuid.uuid4())[:8]
        remote_fasta = f"/tmp/beak_hmmscan_{tag}.fasta"
        remote_domtbl = f"/tmp/beak_hmmscan_{tag}.domtblout"

        try:
            # Upload query
            self.conn.put(fasta_path, remote_fasta)

            # Run hmmscan synchronously
            hmm_db = f"{pfam_path}/{PFAM_HMM_FILE}"
            cmd = (
                f"hmmscan --domtblout {remote_domtbl} --noali "
                f"-E {evalue} {hmm_db} {remote_fasta}"
            )
            result = self.conn.run(cmd, hide=True, warn=True)

            if not result.ok:
                stderr = result.stderr.strip() if result.stderr else ''
                raise RuntimeError(f"hmmscan failed: {stderr}")

            # Read results
            cat_result = self.conn.run(f"cat {remote_domtbl}", hide=True, warn=True)
            if not cat_result.ok:
                return []

            return self._parse_domtblout(cat_result.stdout)

        finally:
            # Clean up temp files
            self.conn.run(
                f"rm -f {remote_fasta} {remote_domtbl}",
                hide=True, warn=True,
            )

    def _parse_domtblout(self, text: str) -> List[Dict]:
        """Parse hmmscan --domtblout output into structured records.

        The domtblout format has 22 whitespace-delimited fields followed
        by a free-text description. Comment lines start with '#'.

        Returns:
            List of dicts sorted by i_evalue (ascending)
        """
        hits = []

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 22 fixed fields + description (field 22+)
            parts = line.split(None, 22)
            if len(parts) < 22:
                continue

            # Strip version suffix from Pfam accession (PF00069.29 -> PF00069)
            accession = parts[1].split('.')[0]

            hits.append({
                'pfam_id': accession,
                'pfam_name': parts[0],
                'description': parts[22] if len(parts) > 22 else '',
                'evalue': float(parts[6]),
                'score': float(parts[7]),
                'bias': float(parts[8]),
                'c_evalue': float(parts[11]),
                'i_evalue': float(parts[12]),
                'hmm_from': int(parts[15]),
                'hmm_to': int(parts[16]),
                'ali_from': int(parts[17]),
                'ali_to': int(parts[18]),
                'env_from': int(parts[19]),
                'env_to': int(parts[20]),
            })

        hits.sort(key=lambda h: h['i_evalue'])
        return hits
