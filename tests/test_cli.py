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
