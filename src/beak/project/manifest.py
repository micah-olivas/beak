"""Read/write beak.project.toml.

Uses tomllib (3.11+) or tomli for reads, tomli-w for writes — the
latter handles datetimes, lists, and arrays-of-tables that the
hand-rolled writer in beak.config doesn't.
"""

import sys
from pathlib import Path
from typing import Any, Dict


def _tomllib():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    try:
        import tomli as tomllib  # type: ignore
        return tomllib
    except ImportError as e:
        raise ImportError(
            "Python <3.11 requires the 'tomli' package. Install with: pip install tomli"
        ) from e


def read_manifest(path: Path) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return _tomllib().load(f)


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    import tomli_w
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        tomli_w.dump(data, f)
