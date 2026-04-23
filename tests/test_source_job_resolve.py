"""Tests for cli/submit.py::_resolve_source_job_fasta dispatcher.

Covers job-type dispatch (search / align / pipeline), unsupported
types, missing jobs, and align-output-format validation. SSH is
never touched — the MMseqsSearch branch is mocked at the class
level so no connection is attempted.
"""

import json
from pathlib import Path

import click
import pytest

from beak.cli import submit as submit_mod


def _write_job_db(home: Path, jobs: dict):
    beak_dir = home / ".beak"
    beak_dir.mkdir(parents=True, exist_ok=True)
    (beak_dir / "jobs.json").write_text(json.dumps(jobs))


class TestResolveSourceJobFasta:
    def test_missing_db_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        with pytest.raises(click.BadParameter, match="No job database"):
            submit_mod._resolve_source_job_fasta('anything')

    def test_unknown_job_id_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {'known_id': {'job_type': 'search'}})
        with pytest.raises(click.BadParameter, match="Unknown job id"):
            submit_mod._resolve_source_job_fasta('not_here')

    def test_unsupported_job_type_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'tax1': {'job_type': 'taxonomy', 'remote_path': '/r/tax1'},
        })
        with pytest.raises(click.BadParameter, match="Can't chain embeddings"):
            submit_mod._resolve_source_job_fasta('tax1')

    def test_search_dispatches_to_manager(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            's1': {'job_type': 'search', 'remote_path': '/r/s1'},
        })

        # Stub MMseqsSearch so no SSH is attempted
        class StubSearch:
            def __init__(self):
                pass
            def ensure_remote_hits_fasta(self, job_id):
                assert job_id == 's1'
                return '/r/s1/hits.fasta'

        import beak.remote.search as search_mod
        monkeypatch.setattr(search_mod, 'MMseqsSearch', StubSearch)

        assert submit_mod._resolve_source_job_fasta('s1') == '/r/s1/hits.fasta'

    def test_align_default_format_is_fasta(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'a1': {
                'job_type': 'align',
                'remote_path': '/r/a1',
                'parameters': {'algorithm': 'clustalo'},
            },
        })
        assert submit_mod._resolve_source_job_fasta('a1') == '/r/a1/alignment.fasta'

    def test_align_non_fasta_format_refuses_with_actionable_msg(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'a1': {
                'job_type': 'align',
                'remote_path': '/r/a1',
                'parameters': {'output_format': 'phylip'},
            },
        })
        with pytest.raises(click.BadParameter, match="phylip"):
            submit_mod._resolve_source_job_fasta('a1')

    def test_align_respects_top_level_output_format_field(
        self, tmp_path, monkeypatch,
    ):
        # Some records stash output_format at the top level rather than
        # inside parameters — the resolver must honor both.
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'a1': {
                'job_type': 'align',
                'remote_path': '/r/a1',
                'output_format': 'nexus',
                'parameters': {},
            },
        })
        with pytest.raises(click.BadParameter, match="nexus"):
            submit_mod._resolve_source_job_fasta('a1')

    def test_pipeline_dispatches_to_pipeline_resolver(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'p1': {
                'job_type': 'pipeline',
                'remote_path': '/r/p1',
                'steps': [
                    {'type': 'search', 'params': {}},
                    {'type': 'align', 'params': {}},
                ],
            },
        })
        path = submit_mod._resolve_source_job_fasta('p1')
        assert path == '/r/p1/02_align/alignment.fasta'

    def test_embedding_source_rejected(self, tmp_path, monkeypatch):
        # Chaining embeddings off another embeddings job doesn't make sense
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        _write_job_db(tmp_path, {
            'e1': {'job_type': 'embeddings', 'remote_path': '/r/e1'},
        })
        with pytest.raises(click.BadParameter, match="Can't chain embeddings"):
            submit_mod._resolve_source_job_fasta('e1')
