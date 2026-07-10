"""Smoke tests for beak CLI using click.testing.CliRunner."""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from beak.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIHelp:
    def test_main_help(self, runner):
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'BEAK' in result.output

    def test_config_help(self, runner):
        result = runner.invoke(main, ['config', '--help'])
        assert result.exit_code == 0
        assert 'init' in result.output

    def test_search_help(self, runner):
        result = runner.invoke(main, ['search', '--help'])
        assert result.exit_code == 0
        assert '--db' in result.output

    def test_jobs_help(self, runner):
        result = runner.invoke(main, ['jobs', '--help'])
        assert result.exit_code == 0
        assert '--type' in result.output

    def test_pfam_help(self, runner):
        result = runner.invoke(main, ['pfam', '--help'])
        assert result.exit_code == 0
        assert '--uniprot' in result.output
        assert '--pfam' in result.output
        assert '--taxonomy' in result.output
        assert '--lineage' in result.output
        assert '--evalue' in result.output

    def test_setup_help(self, runner):
        result = runner.invoke(main, ['setup', '--help'])
        assert result.exit_code == 0
        assert 'pfam' in result.output

    def test_setup_pfam_help(self, runner):
        result = runner.invoke(main, ['setup', 'pfam', '--help'])
        assert result.exit_code == 0
        assert '--system' in result.output
        assert '--status' in result.output
        assert '--update' in result.output

    def test_structures_help(self, runner):
        result = runner.invoke(main, ['structures', '--help'])
        assert result.exit_code == 0
        assert '--source' in result.output
        assert '--selection' in result.output
        assert '--find-only' in result.output
        assert '--output-dir' in result.output


class TestConfigCommands:
    def test_config_show_no_config(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr('beak.config.CONFIG_PATH', tmp_path / 'nonexistent.toml')
        result = runner.invoke(main, ['config', 'show'])
        assert result.exit_code == 0

    def test_config_init(self, runner, tmp_path, monkeypatch):
        config_file = tmp_path / 'config.toml'
        monkeypatch.setattr('beak.config.CONFIG_PATH', config_file)
        monkeypatch.setattr('beak.config.CONFIG_DIR', tmp_path)
        result = runner.invoke(main, ['config', 'init'],
                               input='myserver\nmyuser\n~/.ssh/id_rsa\n~/beak_jobs\n')
        assert result.exit_code == 0
        assert config_file.exists()


class TestJobsCommand:
    def test_jobs_no_db(self, runner, tmp_path, monkeypatch):
        # Patch Path.home() so no jobs.json is found
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        result = runner.invoke(main, ['jobs'])
        assert result.exit_code == 0

    def test_jobs_with_data(self, runner, tmp_path, monkeypatch):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text(json.dumps({
            'abc123': {
                'job_type': 'search',
                'name': 'test_search',
                'status': 'COMPLETED',
                'submitted_at': '2025-01-01T00:00:00',
            }
        }))

        # Patch Path.home() to use tmp_path
        original_home = Path.home

        def fake_home():
            return tmp_path

        monkeypatch.setattr(Path, 'home', staticmethod(fake_home))

        # Also need to create the .beak directory structure
        beak_dir = tmp_path / ".beak"
        beak_dir.mkdir(exist_ok=True)
        (beak_dir / "jobs.json").write_text(jobs_file.read_text())

        result = runner.invoke(main, ['jobs'])
        assert result.exit_code == 0
        assert 'abc123' in result.output
        assert 'test_search' in result.output


class _FakeMgr:
    """Stand-in job manager: no SSH, deterministic ids and terminal state."""

    def __init__(self, final='COMPLETED'):
        self.final = final

    def submit(self, *args, quiet=False, **kwargs):
        # Mirrors the real managers: they print a human line unless quiet.
        if not quiet:
            print("HUMAN confirmation line")
        return "deadbeef"

    def wait(self, job_id, check_interval=30, verbose=True):
        return self.final


@pytest.fixture
def query_fasta(tmp_path):
    fa = tmp_path / "q.fasta"
    fa.write_text(">q\nMKTAYIAKQR\n")
    return str(fa)


class TestMachineOutput:
    """--json / --wait / exit-code contract for agent-driven submission."""

    def _patch(self, monkeypatch, final='COMPLETED'):
        monkeypatch.setattr('beak.cli.submit.get_manager',
                            lambda **k: _FakeMgr(final))

    def test_submit_flags_present_on_all_commands(self, runner):
        for cmd in ('search', 'taxonomy', 'align', 'embeddings'):
            out = runner.invoke(main, [cmd, '--help']).output
            for flag in ('--json', '--wait', '--interval'):
                assert flag in out, f"{flag} missing on {cmd}"

    def test_json_submit_emits_single_clean_object(self, runner, monkeypatch, query_fasta):
        self._patch(monkeypatch)
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'j1', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.output.strip())   # must be exactly one JSON object
        assert obj == {'job_id': 'deadbeef', 'job_type': 'search',
                       'name': 'j1', 'status': 'SUBMITTED'}
        assert 'HUMAN' not in res.output        # quiet suppressed the human line

    def test_group_level_flag_also_works(self, runner, monkeypatch, query_fasta):
        self._patch(monkeypatch)
        res = runner.invoke(main, ['--json', 'search', query_fasta,
                                   '--db', 'uniref90', '--name', 'j2'])
        assert res.exit_code == 0
        assert json.loads(res.output.strip())['name'] == 'j2'

    def test_wait_completed_exits_zero(self, runner, monkeypatch, query_fasta):
        self._patch(monkeypatch, final='COMPLETED')
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'j3', '--json', '--wait'])
        assert res.exit_code == 0
        assert json.loads(res.output.strip())['status'] == 'COMPLETED'

    def test_wait_failed_exits_one(self, runner, monkeypatch, query_fasta):
        self._patch(monkeypatch, final='FAILED')
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'j4', '--json', '--wait'])
        assert res.exit_code == 1                # JobFailed
        line = next(l for l in res.output.splitlines() if l.strip().startswith('{'))
        assert json.loads(line)['status'] == 'FAILED'

    def test_human_mode_unchanged(self, runner, monkeypatch, query_fasta):
        self._patch(monkeypatch)
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'j5'])
        assert res.exit_code == 0
        assert 'HUMAN confirmation line' in res.output
        assert '{' not in res.output             # no JSON leaked into human mode
