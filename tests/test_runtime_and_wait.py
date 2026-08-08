"""Tests for accurate job runtime + wait() connection resilience.

Both fixes live in remote/base.py and are exercised without SSH: the
runtime helpers are pure static/classmethods, and wait() is driven with a
stubbed status() + fake connection.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import beak
from beak.remote.base import RemoteJobManager


class TestParseStatusEpochs:
    def test_both_present(self):
        lines = ['Job started: x', 'STARTED_EPOCH=1000', 'RUNNING',
                 'ENDED_EPOCH=1240', 'COMPLETED']
        assert RemoteJobManager._parse_status_epochs(lines) == (1000, 1240)

    def test_missing_end(self):
        assert RemoteJobManager._parse_status_epochs(
            ['STARTED_EPOCH=1000', 'RUNNING']) == (1000, None)

    def test_none(self):
        assert RemoteJobManager._parse_status_epochs(
            ['Job started', 'RUNNING']) == (None, None)

    def test_garbage_ignored(self):
        assert RemoteJobManager._parse_status_epochs(
            ['STARTED_EPOCH=notanint', 'ENDED_EPOCH=1240']) == (None, 1240)


class TestComputeRuntime:
    def _job(self, ago_seconds):
        submitted = (datetime.now() - timedelta(seconds=ago_seconds)).isoformat()
        return {'submitted_at': submitted}

    def test_epoch_is_true_compute_time_regardless_of_submit(self):
        # Submitted 3h ago (mimics the reconnect-gap bug), but epochs say
        # the job ran 4 minutes — runtime must reflect the 4 minutes.
        job = self._job(ago_seconds=3 * 3600)
        lines = ['STARTED_EPOCH=1000', 'ENDED_EPOCH=1240', 'COMPLETED']
        assert RemoteJobManager._compute_runtime(job, 'COMPLETED', lines) == '0:04:00'

    def test_epoch_ignored_when_end_before_start(self):
        # Clock skew safety: fall back rather than emit a negative duration.
        job = self._job(ago_seconds=300)
        lines = ['STARTED_EPOCH=2000', 'ENDED_EPOCH=1000', 'COMPLETED']
        finished = (datetime.fromisoformat(job['submitted_at'])
                    + timedelta(seconds=300)).isoformat()
        assert RemoteJobManager._compute_runtime(
            job, 'COMPLETED', lines, finished_at=finished) == '0:05:00'

    def test_freeze_for_terminal_without_epochs(self):
        # Old-style job (no markers): freeze at finished_at - submitted_at,
        # NOT now - submitted_at (which would keep growing).
        job = self._job(ago_seconds=99999)
        finished = (datetime.fromisoformat(job['submitted_at'])
                    + timedelta(minutes=5)).isoformat()
        assert RemoteJobManager._compute_runtime(
            job, 'COMPLETED', ['COMPLETED'], finished_at=finished) == '0:05:00'

    def test_running_is_live_since_submit(self):
        job = self._job(ago_seconds=90)
        rt = RemoteJobManager._compute_runtime(job, 'RUNNING', ['RUNNING'])
        # ~90s since submit; tolerate the second boundary.
        assert rt.startswith('0:01:')


class _FakeConn:
    def __init__(self):
        self.opens = 0
        self.closes = 0

    def open(self):
        self.opens += 1

    def close(self):
        self.closes += 1


class TestWaitResilience:
    def _mgr(self):
        # Bare instance: no __init__, so no SSH.
        m = RemoteJobManager.__new__(RemoteJobManager)
        m.conn = _FakeConn()
        return m

    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr('time.sleep', lambda s: None)
        m = self._mgr()
        calls = {'n': 0}

        def flaky_status(job_id):
            calls['n'] += 1
            if calls['n'] <= 2:
                raise OSError(49, "Can't assign requested address")
            return {'status': 'COMPLETED', 'runtime': '0:00:05'}

        m.status = flaky_status
        result = m.wait('job1', check_interval=1, verbose=False)
        assert result == 'COMPLETED'
        assert calls['n'] == 3          # 2 failures + 1 success
        assert m.conn.opens == 2        # reconnected once per failure

    def test_gives_up_after_max(self, monkeypatch):
        monkeypatch.setattr('time.sleep', lambda s: None)
        m = self._mgr()
        m.status = lambda job_id: (_ for _ in ()).throw(OSError("boom"))
        with pytest.raises(OSError):
            m.wait('job1', check_interval=1, verbose=False, max_conn_retries=3)


# Every job-script generator must emit the epoch markers, or terminal jobs
# of that type silently fall back to the (less accurate) freeze path.
_REMOTE = Path(beak.__file__).parent / 'remote'


@pytest.mark.parametrize('module', ['search', 'taxonomy', 'align',
                                     'embeddings', 'pipeline'])
def test_job_script_emits_epoch_markers(module):
    src = (_REMOTE / f'{module}.py').read_text()
    assert 'STARTED_EPOCH=$(date +%s)' in src, f'{module}: missing STARTED_EPOCH'
    assert 'ENDED_EPOCH=$(date +%s)' in src, f'{module}: missing ENDED_EPOCH'
