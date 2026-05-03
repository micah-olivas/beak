"""Tests for beak.config — config file round-trip, set/get."""

import pytest
from pathlib import Path

from beak.config import (
    _write_toml, _read_toml, save_config, load_config,
    set_config_value, get_database_config, CONFIG_PATH,
)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp directory."""
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr('beak.config.CONFIG_PATH', config_file)
    monkeypatch.setattr('beak.config.CONFIG_DIR', tmp_path)
    return config_file


class TestTomlRoundTrip:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.toml"
        data = {
            'connection': {
                'host': 'example.com',
                'user': 'testuser',
                'port': 22,
            }
        }
        _write_toml(path, data)
        result = _read_toml(path)
        assert result['connection']['host'] == 'example.com'
        assert result['connection']['user'] == 'testuser'
        assert result['connection']['port'] == 22

    def test_bool_values(self, tmp_path):
        path = tmp_path / "test.toml"
        data = {'settings': {'verbose': True, 'quiet': False}}
        _write_toml(path, data)
        result = _read_toml(path)
        assert result['settings']['verbose'] is True
        assert result['settings']['quiet'] is False


class TestSaveLoadConfig:
    def test_save_and_load(self, tmp_config):
        config = {
            'connection': {
                'host': 'myserver.edu',
                'user': 'testuser',
            }
        }
        save_config(config)
        loaded = load_config()
        assert loaded['connection']['host'] == 'myserver.edu'

    def test_load_missing_returns_empty(self, tmp_config):
        result = load_config()
        assert result == {}


class TestSetConfigValue:
    def test_set_dotted_key(self, tmp_config):
        save_config({'connection': {'host': 'old'}})
        set_config_value('connection.host', 'new')
        result = load_config()
        assert result['connection']['host'] == 'new'

    def test_set_creates_section(self, tmp_config):
        save_config({})
        set_config_value('connection.host', 'example.com')
        result = load_config()
        assert result['connection']['host'] == 'example.com'


class TestGetDatabaseConfig:
    def test_get_database_config(self, tmp_config):
        save_config({'databases': {'pfam_path': '/srv/pfam'}})
        result = get_database_config()
        assert result['pfam_path'] == '/srv/pfam'

    def test_get_database_config_empty(self, tmp_config):
        save_config({'connection': {'host': 'example.com'}})
        result = get_database_config()
        assert result == {'pfam_path': None}

    def test_get_database_config_no_file(self, tmp_config):
        result = get_database_config()
        assert result == {'pfam_path': None}
