"""Session-based API for sharing a single SSH connection across job types."""

from pathlib import Path
from typing import Optional

from fabric import Connection

from .base import RemoteJobManager


class BeakSession:
    """Single SSH connection shared across all beak job types.

    Usage:
        from beak.remote import BeakSession

        bk = BeakSession()  # reads ~/.beak/config.toml
        job = bk.search.submit("query.fasta", database="uniref90")
        bk.search.wait(job)
        results = bk.search.get_results(job)

        # Same connection for taxonomy
        tax_job = bk.taxonomy.submit("hits.fasta")
    """

    def __init__(self, host: Optional[str] = None, user: Optional[str] = None,
                 key_path: Optional[str] = None, remote_job_dir: Optional[str] = None):
        from ..config import get_default_connection

        defaults = get_default_connection()
        host = host or defaults.get('host')
        user = user or defaults.get('user')
        key_path = key_path or defaults.get('key_path')
        self._remote_job_dir = remote_job_dir or defaults.get('remote_job_dir')

        if not host or not user:
            raise ValueError(
                "host and user are required. Provide them as arguments or "
                "configure defaults with: beak config init"
            )

        if key_path is None:
            key_path = RemoteJobManager._find_ssh_key(None)

        self._conn = Connection(
            host=host,
            user=user,
            connect_timeout=10,
            connect_kwargs={"key_filename": str(Path(key_path).expanduser())}
        )

        # Lazy instances
        self._search = None
        self._taxonomy = None
        self._align = None
        self._embeddings = None
        self._pipeline = None
        self._hmmer = None

    @property
    def search(self):
        """MMseqs2 search manager"""
        if self._search is None:
            from .search import MMseqsSearch
            self._search = MMseqsSearch(connection=self._conn,
                                         remote_job_dir=self._remote_job_dir)
        return self._search

    @property
    def taxonomy(self):
        """MMseqs2 taxonomy manager"""
        if self._taxonomy is None:
            from .taxonomy import MMseqsTaxonomy
            self._taxonomy = MMseqsTaxonomy(connection=self._conn,
                                             remote_job_dir=self._remote_job_dir)
        return self._taxonomy

    @property
    def align(self):
        """Clustal Omega alignment manager"""
        if self._align is None:
            from .align import ClustalAlign
            self._align = ClustalAlign(connection=self._conn,
                                        remote_job_dir=self._remote_job_dir)
        return self._align

    @property
    def embeddings(self):
        """ESM embeddings manager"""
        if self._embeddings is None:
            from .embeddings import ESMEmbeddings
            self._embeddings = ESMEmbeddings(connection=self._conn,
                                              remote_job_dir=self._remote_job_dir)
        return self._embeddings

    @property
    def pipeline(self):
        """Pipeline orchestration manager"""
        if self._pipeline is None:
            from .pipeline import Pipeline
            self._pipeline = Pipeline(connection=self._conn,
                                       remote_job_dir=self._remote_job_dir)
        return self._pipeline

    @property
    def hmmer(self):
        """HMMER scan manager (synchronous Pfam domain search)"""
        if self._hmmer is None:
            from .hmmer import HmmerScan
            self._hmmer = HmmerScan(connection=self._conn)
        return self._hmmer
