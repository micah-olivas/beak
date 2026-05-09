"""Tests for jobs.json corruption-recovery and quarantine paths.

Locks in the regression-pass fixes for `RemoteJobManager._load_job_db`:
- A non-empty corrupt file is preserved at `*.corrupt-<ts>` rather than
  silently overwritten with `{}` on the next save.
- Two corruptions in the same wall-second do NOT collide and lose the
  earlier quarantine (the bug pre-fix used `int(time.time())`).
- A corrupt `.bak` does not trigger silent fall-through to `{}`; it is
  also quarantined.

Each test corresponds to a specific bug the fix addresses. The happy
path is exercised transitively by the recovery scenarios.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from beak.remote.base import RemoteJobManager


class _StubManager(RemoteJobManager):
    """Bypass `__init__` so tests don't open SSH or instantiate Fabric.

    The recovery path only touches `self.local_job_db`, the lock, and
    the static `_quarantine` helper — none of which need a connection.
    """

    def __init__(self, local_job_db: Path) -> None:  # noqa: D401
        self.local_job_db = local_job_db


@pytest.fixture
def mgr(tmp_path):
    return _StubManager(tmp_path / "jobs.json")


def test_corrupt_file_is_quarantined_not_clobbered(mgr):
    """Pre-fix, the corrupt bytes silently became `{}` and the next
    save overwrote them."""
    mgr.local_job_db.write_text("{not valid json")
    result = mgr._load_job_db()
    assert result == {}
    # Original file is gone (renamed away) and a quarantine sibling
    # holds the bytes for forensic recovery.
    assert not mgr.local_job_db.exists()
    siblings = list(mgr.local_job_db.parent.glob("jobs.json.corrupt-*"))
    assert len(siblings) == 1
    assert siblings[0].read_text() == "{not valid json"


def test_two_quarantines_in_same_second_do_not_collide(mgr):
    """The pre-regression-fix quarantine used 1-second resolution and
    `os.replace`, so a second corruption in the same second silently
    overwrote the first quarantine. The fix uses time_ns + a numeric
    suffix on collision."""
    mgr.local_job_db.write_text("{first corrupt")
    mgr._load_job_db()  # quarantines the file
    first_q = list(mgr.local_job_db.parent.glob("jobs.json.corrupt-*"))
    assert len(first_q) == 1
    first_bytes = first_q[0].read_text()

    # Force time_ns to return a constant so we exercise the
    # collision-resolution branch directly.
    mgr.local_job_db.write_text("{second corrupt")
    fixed_ns = 1_700_000_000_000_000_000
    with patch("time.time_ns", return_value=fixed_ns):
        mgr._load_job_db()

    quarantines = sorted(mgr.local_job_db.parent.glob("jobs.json.corrupt-*"))
    # Both quarantines preserved — the bug would have left only one.
    assert len(quarantines) == 2
    contents = {p.read_text() for p in quarantines}
    assert contents == {first_bytes, "{second corrupt"}


def test_bak_recovery_when_main_corrupt(mgr):
    """End-to-end: a torn write to the main file is transparently
    recovered from the .bak written by the previous successful save."""
    mgr._save_job_db({"v1": {"status": "RUNNING"}})
    mgr._save_job_db({"v2": {"status": "DONE"}})  # writes .bak from v1
    mgr.local_job_db.write_text("{torn write")
    recovered = mgr._load_job_db()
    # .bak holds the most recent successful save before the torn one.
    assert "v1" in recovered


def test_corrupt_bak_is_also_quarantined(mgr):
    """Pre-fix, a corrupt .bak silently fell through to `{}` and the
    bytes were lost. The fix quarantines the .bak too."""
    mgr.local_job_db.write_text("{main corrupt")
    bak = mgr.local_job_db.with_suffix(mgr.local_job_db.suffix + ".bak")
    bak.write_text("{bak also corrupt")

    assert mgr._load_job_db() == {}
    # Both originals preserved as quarantines, neither lost.
    main_q = list(mgr.local_job_db.parent.glob("jobs.json.corrupt-*"))
    bak_q = list(mgr.local_job_db.parent.glob("jobs.json.bak.bak-corrupt-*"))
    assert len(main_q) == 1
    assert len(bak_q) == 1
    assert main_q[0].read_text() == "{main corrupt"
    assert bak_q[0].read_text() == "{bak also corrupt"
