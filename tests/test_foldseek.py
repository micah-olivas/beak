"""Tests for remote foldseek structural search.

Pure unit tests: no SSH, no foldseek binary, no network. Cover the m8
parser (the format contract), the config getter, and the RemoteFoldseek
pure helpers — param formatting, remote-script building, and database
alias resolution (via a bare instance so no connection is opened).
"""

import pandas as pd
import pytest

from beak.config import get_foldseek_config, save_config
from beak.structures.foldseek import DEFAULT_OUTPUT_COLUMNS, parse_foldseek_m8
from beak.remote.foldseek import FOLDSEEK_DATABASES, RemoteFoldseek


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp directory."""
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr('beak.config.CONFIG_PATH', config_file)
    monkeypatch.setattr('beak.config.CONFIG_DIR', tmp_path)
    return config_file


_SAMPLE_M8 = (
    "query\t1abc_A\t0.812\t210\t30\t2\t1\t210\t5\t214\t"
    "1.2e-40\t512\t0.94\t0.88\t0.98\t0.95\n"
    "query\t2xyz_B\t0.410\t180\t95\t5\t3\t183\t10\t190\t"
    "3.4e-08\t120\t0.61\t0.55\t0.85\t0.80\n"
)


class TestParseM8:
    def test_columns_and_row_count(self):
        df = parse_foldseek_m8(_SAMPLE_M8)
        assert list(df.columns) == DEFAULT_OUTPUT_COLUMNS
        assert len(df) == 2

    def test_numeric_typing(self):
        df = parse_foldseek_m8(_SAMPLE_M8)
        assert df.loc[0, "target"] == "1abc_A"
        assert df.loc[0, "fident"] == pytest.approx(0.812)
        assert df.loc[0, "alntmscore"] == pytest.approx(0.94)
        assert df.loc[0, "evalue"] == pytest.approx(1.2e-40)
        assert df.loc[0, "lddt"] == pytest.approx(0.88)
        assert df["alnlen"].dtype == "Int64"
        assert int(df.loc[0, "alnlen"]) == 210

    def test_empty_input_yields_empty_typed_frame(self):
        df = parse_foldseek_m8("")
        assert len(df) == 0
        assert list(df.columns) == DEFAULT_OUTPUT_COLUMNS

    def test_blank_lines_skipped(self):
        df = parse_foldseek_m8("\n\n" + _SAMPLE_M8 + "\n")
        assert len(df) == 2

    def test_custom_columns(self):
        df = parse_foldseek_m8("query\t1abc_A\t0.9\n",
                               columns=["query", "target", "alntmscore"])
        assert list(df.columns) == ["query", "target", "alntmscore"]
        assert df.loc[0, "alntmscore"] == pytest.approx(0.9)

    def test_short_row_is_padded(self):
        df = parse_foldseek_m8("query\t1abc_A\n")
        assert len(df) == 1
        assert df.loc[0, "target"] == "1abc_A"
        assert pd.isna(df.loc[0, "alntmscore"])

    def test_extra_fields_truncated(self):
        row = _SAMPLE_M8.splitlines()[0] + "\textra\tfields"
        df = parse_foldseek_m8(row + "\n")
        assert list(df.columns) == DEFAULT_OUTPUT_COLUMNS
        assert len(df) == 1


class TestGetFoldseekConfig:
    def test_reads_keys(self, tmp_config):
        save_config({'foldseek': {'db_path': '/srv/fs/pdb/db', 'db_name': 'pdb'}})
        assert get_foldseek_config() == {
            'db_path': '/srv/fs/pdb/db', 'db_name': 'pdb',
        }

    def test_empty_when_unset(self, tmp_config):
        save_config({'connection': {'host': 'example.com'}})
        assert get_foldseek_config() == {'db_path': None, 'db_name': None}

    def test_no_file(self, tmp_config):
        assert get_foldseek_config() == {'db_path': None, 'db_name': None}


class TestFormatParams:
    def test_single_and_multi_char_flags(self):
        s = RemoteFoldseek._format_params({'s': 9.5, 'e': 0.001, 'max_seqs': 1000})
        assert '-s 9.5' in s
        assert '-e 0.001' in s
        assert '--max-seqs 1000' in s

    def test_empty(self):
        assert RemoteFoldseek._format_params({}) == ''


class TestBuildJobScript:
    def test_contains_easy_search_and_status_markers(self):
        script = RemoteFoldseek._build_job_script(
            '/home/u/beak_jobs/abc123',
            '/home/u/beak_jobs/abc123/query.cif',
            '/srv/fs/pdb/db',
            '-s 9.5 --threads 8',
        )
        assert 'foldseek easy-search' in script
        assert '/home/u/beak_jobs/abc123/query.cif' in script
        assert '/srv/fs/pdb/db' in script
        assert '/home/u/beak_jobs/abc123/results.m8' in script
        assert "echo 'RUNNING'" in script
        assert 'COMPLETED' in script and 'FAILED' in script
        assert '--format-output' in script
        # pipefail ensures foldseek's exit (not tee's) drives the marker.
        assert 'set -o pipefail' in script
        assert '-s 9.5 --threads 8' in script

    def test_format_output_columns_match_parser(self):
        script = RemoteFoldseek._build_job_script('p', 'q.cif', 'db', '')
        assert ",".join(DEFAULT_OUTPUT_COLUMNS) in script


class TestDatabaseResolution:
    def _bare(self):
        # Construct without __init__ so no SSH connection is opened; the
        # methods under test only touch class attributes.
        return RemoteFoldseek.__new__(RemoteFoldseek)

    def test_available_dbs_derived_from_registry(self):
        assert set(RemoteFoldseek.AVAILABLE_DBS) == set(FOLDSEEK_DATABASES)
        for alias, (_, subdir) in FOLDSEEK_DATABASES.items():
            assert RemoteFoldseek.AVAILABLE_DBS[alias] == f"{subdir}/db"

    def test_alias_resolves_under_db_base(self):
        mgr = self._bare()
        resolved = mgr._resolve_database('pdb')
        assert resolved == f"{RemoteFoldseek.DB_BASE_PATH}/foldseek/pdb/db"

    def test_absolute_path_passthrough(self):
        mgr = self._bare()
        assert mgr._resolve_database('/data/custom/db') == '/data/custom/db'

    def test_job_type(self):
        assert RemoteFoldseek.JOB_TYPE == 'foldseek'
