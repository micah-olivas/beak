"""Configuration management for beak.

Reads/writes ~/.beak/config.toml for persistent settings like
default SSH connection parameters.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

CONFIG_DIR = Path.home() / ".beak"
CONFIG_PATH = CONFIG_DIR / "config.toml"


def _read_toml(path: Path) -> Dict:
    """Read a TOML file, using tomllib (3.11+) or tomli fallback."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError(
                "Python <3.11 requires the 'tomli' package for TOML support. "
                "Install with: pip install tomli"
            )

    with open(path, 'rb') as f:
        return tomllib.load(f)


def _write_toml(path: Path, data: Dict):
    """Write a dict as TOML. Uses simple manual formatting to avoid extra deps."""
    lines = []
    for section, values in data.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for key, val in values.items():
                if isinstance(val, str):
                    lines.append(f'{key} = "{val}"')
                elif isinstance(val, bool):
                    lines.append(f'{key} = {"true" if val else "false"}')
                elif isinstance(val, (int, float)):
                    lines.append(f'{key} = {val}')
                else:
                    lines.append(f'{key} = "{val}"')
            lines.append("")
        else:
            if isinstance(values, str):
                lines.append(f'{section} = "{values}"')
            else:
                lines.append(f'{section} = {values}')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_config() -> Dict:
    """Load configuration from ~/.beak/config.toml.

    Returns empty dict if config file doesn't exist.
    """
    if not CONFIG_PATH.exists():
        return {}
    return _read_toml(CONFIG_PATH)


def save_config(config: Dict):
    """Save configuration to ~/.beak/config.toml."""
    _write_toml(CONFIG_PATH, config)


def get_default_connection() -> Dict:
    """Get default connection kwargs from config.

    Returns:
        Dict with keys: host, user, key_path, remote_job_dir (any may be None)
    """
    config = load_config()
    conn = config.get('connection', {})
    return {
        'host': conn.get('host'),
        'user': conn.get('user'),
        'key_path': conn.get('key_path'),
        'remote_job_dir': conn.get('remote_job_dir'),
    }


def get_database_config() -> Dict:
    """Get database path configuration from config.

    Returns:
        Dict with keys: pfam_path (may be None)
    """
    config = load_config()
    dbs = config.get('databases', {})
    return {
        'pfam_path': dbs.get('pfam_path'),
    }


def get_foldseek_config() -> Dict:
    """Get local foldseek configuration from config.

    Foldseek runs locally (not on the remote), so unlike ``pfam_path``
    these point at the *local* filesystem.

    Returns:
        Dict with keys: binary (path to the foldseek executable, or None
        to fall back to $PATH), db_path (target database prefix), db_name
        (which prebuilt database db_path holds, for display). Any may be
        None.
    """
    config = load_config()
    fs = config.get('foldseek', {})
    return {
        'binary': fs.get('binary'),
        'db_path': fs.get('db_path'),
        'db_name': fs.get('db_name'),
    }


def get_docker_config() -> Dict:
    """Get Docker service configuration from config.

    Returns:
        Dict with keys:
          service_dir:  absolute remote path to the shared Docker service
                        directory. If set, all beak users on this remote point
                        at the same docker-compose project and share a single
                        running container per service. If None, each user
                        deploys its own copy under `{remote_job_dir}/docker`.
          project_name: docker compose project name (default: 'beak') — kept
                        consistent across users so `docker compose` commands
                        scope to the same service regardless of working dir.
    """
    config = load_config()
    d = config.get('docker', {})
    return {
        'service_dir': d.get('service_dir'),
        'project_name': d.get('project_name', 'beak'),
    }


def get_compute_threads(default: int = 8) -> int:
    """Default thread cap for remote MMseqs2 jobs.

    MMseqs2 with no `--threads` defaults to every core on the box,
    which is a problem on a shared remote server. Read
    ``compute.threads`` from config; fall back to ``default`` (8) so
    a fresh user doesn't hog the server before they've configured
    anything. Per-job overrides via ``mmseqs_params={'threads': N}``
    still win — this is only the default.
    """
    config = load_config()
    val = (config.get('compute') or {}).get('threads')
    try:
        n = int(val)
        return n if n >= 1 else default
    except (TypeError, ValueError):
        return default


def config_exists() -> bool:
    """Check if a config file exists."""
    return CONFIG_PATH.exists()


def set_config_value(dotted_key: str, value: str):
    """Set a config value using dotted notation (e.g., 'connection.host').

    Args:
        dotted_key: Key in 'section.key' format
        value: Value to set
    """
    config = load_config()
    parts = dotted_key.split('.', 1)

    if len(parts) == 2:
        section, key = parts
        if section not in config:
            config[section] = {}
        config[section][key] = value
    else:
        config[parts[0]] = value

    save_config(config)
