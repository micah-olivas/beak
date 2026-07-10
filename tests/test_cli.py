"""Smoke tests for beak CLI using click.testing.CliRunner."""

import json
import sys
import pytest
from pathlib import Path
from click.testing import CliRunner

from beak.cli import main, cli_entry
from beak.remote.base import _parse_load_probe


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


class _StatusMgr:
    def status(self, job_id):
        print("MANAGER CHATTER")   # simulates a manager raw print
        return {'job_id': job_id, 'name': 'j', 'status': 'RUNNING',
                'runtime': '0:01:00', 'job_type': 'search'}


class _SearchResultsMgr:
    JOB_TYPE = 'search'

    def get_results(self, job_id, parse=False):
        print("✓ Downloaded 164 hit sequences")   # manager chatter
        return "/proj/hits.fasta"

    def get_project_dir(self, job_id):
        return "/proj"


def _split_runner():
    """CliRunner with stdout/stderr separated, across Click versions."""
    try:
        return CliRunner(mix_stderr=False)     # Click < 8.2
    except TypeError:
        return CliRunner()                     # Click >= 8.2 (already separate)


def _stderr(res):
    try:
        return res.stderr
    except (ValueError, Exception):
        return ""


class TestMonitorJson:
    """--json for the read commands: status / results / jobs."""

    def test_status_json_emits_flat_object(self, monkeypatch):
        monkeypatch.setattr('beak.cli.jobs.get_manager', lambda **k: _StatusMgr())
        res = _split_runner().invoke(main, ['status', 'abc12345', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.stdout.strip())
        assert obj['status'] == 'RUNNING' and obj['job_id'] == 'abc12345'

    def test_status_json_keeps_chatter_off_stdout(self, monkeypatch):
        monkeypatch.setattr('beak.cli.jobs.get_manager', lambda **k: _StatusMgr())
        res = _split_runner().invoke(main, ['status', 'abc12345', '--json'])
        assert 'CHATTER' not in res.stdout       # stdout is pure JSON
        assert 'CHATTER' in _stderr(res)         # chatter routed to stderr

    def test_results_json_reports_path_not_preview(self, monkeypatch):
        monkeypatch.setattr('beak.cli.jobs.get_manager', lambda **k: _SearchResultsMgr())
        res = _split_runner().invoke(main, ['results', 'abc12345', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.stdout.strip())
        assert obj == {'job_id': 'abc12345', 'job_type': 'search',
                       'results_path': '/proj/hits.fasta'}
        assert 'Downloaded' not in res.stdout    # manager chatter not on stdout

    def test_jobs_json_emits_array(self, tmp_path, monkeypatch):
        beak_dir = tmp_path / ".beak"
        beak_dir.mkdir()
        (beak_dir / "jobs.json").write_text(json.dumps({
            "abc12345": {"job_type": "search", "name": "s1",
                         "status": "COMPLETED",
                         "submitted_at": "2026-07-09T10:00:00"},
        }))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        res = _split_runner().invoke(main, ['jobs', '--no-refresh', '--json'])
        assert res.exit_code == 0
        arr = json.loads(res.stdout.strip())
        assert isinstance(arr, list) and arr[0]['id'] == 'abc12345'
        assert arr[0]['status'] == 'COMPLETED'

    def test_jobs_json_empty_when_no_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        res = _split_runner().invoke(main, ['jobs', '--json'])
        assert res.exit_code == 0
        assert json.loads(res.stdout.strip()) == []


class TestErrorJson:
    """cli_entry() renders errors as JSON on stdout in --json mode."""

    def test_error_as_json_on_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, 'argv',
                            ['beak', 'search', '/no/such.fasta',
                             '--db', 'uniref90', '--json'])
        with pytest.raises(SystemExit) as ei:
            cli_entry()
        assert ei.value.code == 2                       # usage error
        obj = json.loads(capsys.readouterr().out.strip())
        assert obj['exit_code'] == 2
        assert 'File not found' in obj['error']

    def test_error_stays_human_without_json(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, 'argv',
                            ['beak', 'search', '/no/such.fasta', '--db', 'uniref90'])
        with pytest.raises(SystemExit) as ei:
            cli_entry()
        assert ei.value.code == 2
        cap = capsys.readouterr()
        assert cap.out.strip() == ''                    # nothing on stdout
        assert 'File not found' in cap.err              # message on stderr

    def test_success_exits_zero(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['beak', '--help'])
        with pytest.raises(SystemExit) as ei:
            cli_entry()
        assert ei.value.code == 0


class TestDryRun:
    """--dry-run previews the plan without connecting or submitting."""

    def test_search_dry_run_json(self, runner, query_fasta, monkeypatch):
        # If it tried to submit it would need get_manager (a remote); make
        # that explode so the test proves dry-run never reaches it.
        monkeypatch.setattr('beak.cli.submit.get_manager',
                            lambda **k: (_ for _ in ()).throw(AssertionError("connected!")))
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--preset', 'broad', '--dry-run', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.output.strip())
        assert obj['dry_run'] is True
        assert obj['job_type'] == 'search'
        assert obj['database'] == 'uniref90' and obj['preset'] == 'broad'

    def test_embeddings_dry_run_reports_size_estimate(self, runner, query_fasta, monkeypatch):
        monkeypatch.setattr('beak.cli.submit.get_manager',
                            lambda **k: (_ for _ in ()).throw(AssertionError("connected!")))
        res = runner.invoke(main, ['embeddings', query_fasta,
                                   '-m', 'esm2_t33_650M_UR50D',
                                   '--layer', '30,33', '--dry-run', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.output.strip())
        assert obj['dry_run'] is True and obj['job_type'] == 'embeddings'
        assert obj['layers'] == [30, 33]
        assert obj['estimated_output_bytes'] > 0

    def test_dry_run_human_form(self, runner, query_fasta):
        res = runner.invoke(main, ['align', query_fasta, '--dry-run'])
        assert res.exit_code == 0
        assert 'DRY RUN' in res.output
        assert 'algorithm: clustalo' in res.output


class TestLoadProbeParsing:
    """_parse_load_probe: raw probe stdout -> structured load dict."""

    def test_full(self):
        load = _parse_load_probe(
            "0.80 0.60 0.50 2/300 12345",   # /proc/loadavg
            "8",                            # nproc
            "16000 4000",                   # free -m: total available
            "10, 2048, 40960\n95, 39000, 40960",  # 2 GPUs
        )
        assert load['load_1m'] == 0.8 and load['load_15m'] == 0.5
        assert load['n_cpus'] == 8 and load['load_per_cpu'] == 0.1
        assert load['mem_total_mb'] == 16000 and load['mem_available_mb'] == 4000
        assert len(load['gpus']) == 2
        assert load['gpus'][1]['util_pct'] == 95.0
        assert load['gpus'][0]['mem_total_mb'] == 40960

    def test_cpu_only_no_gpu(self):
        load = _parse_load_probe("1.0 1.0 1.0", "4", "", "")
        assert load['load_per_cpu'] == 0.25
        assert 'gpus' not in load and 'mem_total_mb' not in load

    def test_all_empty(self):
        assert _parse_load_probe("", "", "", "") == {}

    def test_garbage_is_omitted_not_guessed(self):
        assert _parse_load_probe("not a number", "abc", "x y", "bad,csv") == {}


class _DoctorConn:
    host = 'srv.example'


class _DoctorMgr:
    def __init__(self, ok=True):
        self._ok = ok
        self.conn = _DoctorConn()

    def verify_remote(self, verbose=False):
        return {'ok': self._ok, 'tools': {}, 'databases': {}, 'disk': {},
                'load': {'load_1m': 0.8, 'n_cpus': 8, 'load_per_cpu': 0.1,
                         'mem_total_mb': 16000, 'mem_available_mb': 4000}}


def _no_pfam(conn):
    raise FileNotFoundError()


class TestDoctorJson:
    def test_includes_load_and_pfam(self, monkeypatch):
        monkeypatch.setattr('beak.cli._common.get_manager', lambda **k: _DoctorMgr(True))
        monkeypatch.setattr('beak.remote.hmmer.resolve_pfam_path', _no_pfam)
        res = _split_runner().invoke(main, ['doctor', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.stdout.strip())
        assert obj['ok'] is True
        assert obj['load']['load_per_cpu'] == 0.1
        assert obj['pfam'] == {'installed': False, 'path': None}

    def test_exit_nonzero_when_not_ok(self, monkeypatch):
        monkeypatch.setattr('beak.cli._common.get_manager', lambda **k: _DoctorMgr(False))
        monkeypatch.setattr('beak.remote.hmmer.resolve_pfam_path', _no_pfam)
        res = _split_runner().invoke(main, ['doctor', '--json'])
        assert res.exit_code == 1
        assert json.loads(res.stdout.strip())['ok'] is False


def _no_submit(**k):
    raise AssertionError("submitted despite an available reuse match")


class TestReuse:
    """--reuse returns an existing non-failed job instead of resubmitting."""

    def test_fingerprint_stable_and_param_sensitive(self, query_fasta):
        from beak.cli._common import job_fingerprint
        a = job_fingerprint('search', query_fasta, {'database': 'uniref90', 'preset': 'default'})
        b = job_fingerprint('search', query_fasta, {'database': 'uniref90', 'preset': 'default'})
        c = job_fingerprint('search', query_fasta, {'database': 'uniref50', 'preset': 'default'})
        assert a == b                       # same inputs -> same fingerprint
        assert a != c                       # a changed param -> different
        assert len(a) == 16

    def _seed(self, tmp_path, query_fasta, status):
        from beak.cli._common import job_fingerprint
        fp = job_fingerprint('search', query_fasta,
                             {'database': 'uniref90', 'preset': 'default'})
        beak_dir = tmp_path / '.beak'
        beak_dir.mkdir()
        (beak_dir / 'jobs.json').write_text(json.dumps({
            'aaaa1111': {'job_type': 'search', 'name': 'prev', 'status': status,
                         'submitted_at': '2026-07-09T09:00:00', 'fingerprint': fp},
        }))

    def test_reuse_hit_returns_existing(self, runner, query_fasta, tmp_path, monkeypatch):
        self._seed(tmp_path, query_fasta, 'RUNNING')
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        monkeypatch.setattr('beak.cli.submit.get_manager', _no_submit)
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'x', '--reuse', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.output.strip())
        assert obj['reused'] is True
        assert obj['job_id'] == 'aaaa1111' and obj['status'] == 'RUNNING'

    def test_failed_job_is_not_reused(self, runner, query_fasta, tmp_path, monkeypatch):
        self._seed(tmp_path, query_fasta, 'FAILED')
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        # A FAILED match must NOT be reused, so it proceeds to submit -> our
        # stub raises, proving no reuse happened.
        monkeypatch.setattr('beak.cli.submit.get_manager',
                            lambda **k: (_ for _ in ()).throw(RuntimeError("would submit")))
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'x', '--reuse', '--json'])
        assert res.exit_code != 0
        assert 'reused' not in (res.output or '')

    def test_no_reuse_flag_ignores_match(self, runner, query_fasta, tmp_path, monkeypatch):
        self._seed(tmp_path, query_fasta, 'RUNNING')
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
        # Without --reuse, an existing match is ignored and it submits.
        monkeypatch.setattr('beak.cli.submit.get_manager',
                            lambda **k: (_ for _ in ()).throw(RuntimeError("would submit")))
        res = runner.invoke(main, ['search', query_fasta, '--db', 'uniref90',
                                   '--name', 'x', '--json'])
        assert res.exit_code != 0            # tried to submit, no reuse


class TestProjectJson:
    """--json for the project management commands."""

    def _projects_dir(self, tmp_path, monkeypatch):
        d = tmp_path / 'projects'
        d.mkdir()
        monkeypatch.setattr('beak.project.project.PROJECTS_DIR', d)
        return d

    def test_list_empty_is_json_array(self, tmp_path, monkeypatch):
        self._projects_dir(tmp_path, monkeypatch)
        res = _split_runner().invoke(main, ['project', 'list', '--json'])
        assert res.exit_code == 0
        assert json.loads(res.stdout.strip()) == []

    def test_init_list_status_roundtrip(self, tmp_path, monkeypatch):
        self._projects_dir(tmp_path, monkeypatch)
        fa = tmp_path / 't.fasta'
        fa.write_text('>t\nMKTAYIAKQRQISFVK\n')

        res = _split_runner().invoke(
            main, ['project', 'init', 'myproj', '--sequence', str(fa), '--json'])
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout.strip())['name'] == 'myproj'

        res = _split_runner().invoke(main, ['project', 'list', '--json'])
        names = [p['name'] for p in json.loads(res.stdout.strip())]
        assert 'myproj' in names

        res = _split_runner().invoke(main, ['project', 'status', 'myproj', '--json'])
        st = json.loads(res.stdout.strip())
        assert st['name'] == 'myproj'
        assert st['layers']['target']['present'] is True
        assert st['layers']['homologs']['present'] is False


class TestLogJson:
    def test_log_json_envelope(self, monkeypatch):
        class _LogMgr:
            def get_log(self, job_id, lines=50):
                print("Files in job directory:")
                print("mmseqs.log (last 50 lines):\nstarted\nCOMPLETED")

        monkeypatch.setattr('beak.cli.jobs.get_manager', lambda **k: _LogMgr())
        res = _split_runner().invoke(main, ['log', 'abc12345', '--json'])
        assert res.exit_code == 0
        obj = json.loads(res.stdout.strip())
        assert obj['job_id'] == 'abc12345' and obj['lines'] == 50
        assert 'COMPLETED' in obj['log']
