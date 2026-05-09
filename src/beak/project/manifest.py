"""Read/write beak.project.toml.

Uses tomllib (3.11+) or tomli for reads, tomli-w for writes — the
latter handles datetimes, lists, and arrays-of-tables that the
hand-rolled writer in beak.config doesn't.
"""

import os
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
    """Atomically write the manifest TOML, with cross-process locking.

    The manifest is the project's single source of truth for job IDs,
    layer pointers, and active-set bookkeeping; a torn write (crash /
    OOM mid-`tomli_w.dump`) leaves invalid TOML on disk and silently
    orphans every layer the file pointed at. Write to a sibling temp
    file, fsync the bytes to disk, then `os.replace` for a POSIX
    atomic rename — readers either see the previous file or the new
    one, never a half-flushed mix.

    A sibling `<file>.lock` is held with `fcntl.flock` for the
    duration of the write. Without it, two processes (e.g. two TUI
    instances on the same project, or a CLI tool writing while the
    TUI is updating) could:
        1. Both write to `<file>.tmp` (last writer wins on the temp).
        2. Both `os.replace` — the second clobbers the first's data.
    flock serializes the whole read-modify-write so concurrent
    callers see one another's writes instead of silently losing
    half. Fabric / fcntl is POSIX-only — beak runs on macOS and
    Linux, and `import fcntl` is fine on both.
    """
    import fcntl
    import tomli_w
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + '.lock')
    tmp = path.with_suffix(path.suffix + '.tmp')
    # Open the lockfile for writing so flock has a real fd to attach
    # to even on the first run. Hold the lock until after replace —
    # flock is released on close, so a context manager is the right
    # shape here.
    with open(lock_path, 'a+') as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            with open(tmp, 'wb') as f:
                tomli_w.dump(data, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
