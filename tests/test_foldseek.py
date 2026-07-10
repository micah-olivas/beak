"""Tests for local foldseek support.

Pure unit tests: no foldseek binary, no subprocess, no network. Cover the
m8 parser (the format contract with foldseek), the config getter, and
binary resolution's fallback chain (monkeypatched ``shutil.which``).
"""

import pandas as pd
import pytest

from beak.config import get_foldseek_config, save_config
from beak.structures.foldseek import (
    DEFAULT_OUTPUT_COLUMNS,
    FoldseekError,
    parse_foldseek_m8,
    resolve_foldseek_binary,
)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp directory."""
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr('beak.config.CONFIG_PATH', config_file)
    monkeypatch.setattr('beak.config.CONFIG_DIR', tmp_path)
    return config_file


# A couple of realistic foldseek easy-search rows in the default column order.
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
        # Integer columns land as nullable Int64.
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
        text = "query\t1abc_A\t0.9\n"
        df = parse_foldseek_m8(text, columns=["query", "target", "alntmscore"])
        assert list(df.columns) == ["query", "target", "alntmscore"]
        assert df.loc[0, "alntmscore"] == pytest.approx(0.9)

    def test_short_row_is_padded(self):
        # Fewer fields than columns -> padded with NA, not dropped.
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
    def test_reads_all_keys(self, tmp_config):
        save_config({'foldseek': {
            'binary': '/opt/foldseek',
            'db_path': '/data/fs/db',
            'db_name': 'PDB',
        }})
        cfg = get_foldseek_config()
        assert cfg == {
            'binary': '/opt/foldseek',
            'db_path': '/data/fs/db',
            'db_name': 'PDB',
        }

    def test_empty_when_unset(self, tmp_config):
        save_config({'connection': {'host': 'example.com'}})
        assert get_foldseek_config() == {
            'binary': None, 'db_path': None, 'db_name': None,
        }

    def test_no_file(self, tmp_config):
        assert get_foldseek_config() == {
            'binary': None, 'db_path': None, 'db_name': None,
        }


class TestResolveBinary:
    def test_explicit_on_path(self, monkeypatch):
        monkeypatch.setattr(
            'shutil.which',
            lambda name: '/usr/bin/foldseek' if name == 'myfs' else None,
        )
        assert resolve_foldseek_binary('myfs') == '/usr/bin/foldseek'

    def test_explicit_missing_raises(self, monkeypatch):
        monkeypatch.setattr('shutil.which', lambda name: None)
        with pytest.raises(FoldseekError):
            resolve_foldseek_binary('/nope/foldseek')

    def test_config_binary_used(self, tmp_config, monkeypatch):
        save_config({'foldseek': {'binary': '/opt/fs/foldseek'}})
        monkeypatch.setattr(
            'shutil.which',
            lambda name: '/opt/fs/foldseek' if name == '/opt/fs/foldseek'
            else None,
        )
        assert resolve_foldseek_binary() == '/opt/fs/foldseek'

    def test_path_fallback(self, tmp_config, monkeypatch):
        save_config({})
        monkeypatch.setattr(
            'shutil.which',
            lambda name: '/usr/local/bin/foldseek' if name == 'foldseek'
            else None,
        )
        assert resolve_foldseek_binary() == '/usr/local/bin/foldseek'

    def test_not_found_raises(self, tmp_config, monkeypatch):
        save_config({})
        monkeypatch.setattr('shutil.which', lambda name: None)
        with pytest.raises(FoldseekError):
            resolve_foldseek_binary()
