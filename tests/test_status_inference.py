"""Unit tests for RemoteJobManager._infer_status — pure, no SSH."""

from datetime import datetime, timedelta

import pytest

from beak.remote.base import RemoteJobManager


def _mgr():
    # Bypass __init__ (would need Fabric + config); _infer_status only
    # reads self._LAUNCH_GRACE_SECONDS which is a class attribute.
    return RemoteJobManager.__new__(RemoteJobManager)


def _job(submitted_seconds_ago=10):
    submitted_at = datetime.now() - timedelta(seconds=submitted_seconds_ago)
    return {'submitted_at': submitted_at.isoformat()}


class TestInferStatus:
    def test_completed_marker_wins(self):
        mgr = _mgr()
        assert mgr._infer_status(_job(), is_running=False,
                                 status_lines=['RUNNING', 'COMPLETED']) == 'COMPLETED'

    def test_failed_marker_wins(self):
        mgr = _mgr()
        assert mgr._infer_status(_job(), is_running=True,
                                 status_lines=['RUNNING', 'FAILED']) == 'FAILED'

    def test_cancelled_marker_overrides_running(self):
        # Cancel appends `CANCELLED` to a status.txt that already has
        # `RUNNING` from the launch script. Without the terminal
        # marker winning, the post-cancel poll keeps reporting
        # RUNNING and the layers panel never updates.
        mgr = _mgr()
        assert mgr._infer_status(
            _job(), is_running=False,
            status_lines=['RUNNING', 'CANCELLED']
        ) == 'CANCELLED'

    def test_dead_pid_with_running_marker_within_grace_is_submitted(self):
        # Dead PID + status.txt = RUNNING + within grace window: the
        # script may simply be booting and hasn't touched the file
        # since the initial RUNNING line. Keep it SUBMITTED.
        mgr = _mgr()
        assert mgr._infer_status(
            _job(submitted_seconds_ago=10), is_running=False,
            status_lines=['RUNNING']
        ) == 'SUBMITTED'

    def test_dead_pid_with_running_marker_past_grace_is_failed(self):
        # The pre-fix bug: PID dead + stale RUNNING marker would be
        # reported as RUNNING forever, because the old logic trusted
        # the file even when the process was gone. After kill -9,
        # OOM, or a server reboot the script never writes FAILED, so
        # we have to infer it from "past grace + dead PID + no
        # terminal marker."
        mgr = _mgr()
        assert mgr._infer_status(
            _job(submitted_seconds_ago=300), is_running=False,
            status_lines=['RUNNING']
        ) == 'FAILED'

    def test_running_from_live_pid(self):
        mgr = _mgr()
        assert mgr._infer_status(_job(), is_running=True,
                                 status_lines=['NO_STATUS']) == 'RUNNING'

    def test_no_status_within_grace_window_is_submitted(self):
        mgr = _mgr()
        # Just submitted, no status.txt yet, PID not running either
        assert mgr._infer_status(_job(submitted_seconds_ago=5),
                                 is_running=False,
                                 status_lines=['NO_STATUS']) == 'SUBMITTED'

    def test_no_status_past_grace_window_is_failed(self):
        mgr = _mgr()
        # More than _LAUNCH_GRACE_SECONDS ago, PID dead, no status file →
        # the launch script itself crashed.
        assert mgr._infer_status(_job(submitted_seconds_ago=300),
                                 is_running=False,
                                 status_lines=['NO_STATUS']) == 'FAILED'

    def test_no_status_past_grace_but_running_stays_running(self):
        mgr = _mgr()
        # PID still alive, even past grace — it's running, just hasn't
        # flushed status.txt yet.
        assert mgr._infer_status(_job(submitted_seconds_ago=300),
                                 is_running=True,
                                 status_lines=['NO_STATUS']) == 'RUNNING'

    def test_malformed_submitted_at_falls_back_to_submitted(self):
        mgr = _mgr()
        job = {'submitted_at': 'not-a-date'}
        assert mgr._infer_status(job, is_running=False,
                                 status_lines=['NO_STATUS']) == 'SUBMITTED'

    def test_missing_submitted_at_falls_back_to_submitted(self):
        mgr = _mgr()
        assert mgr._infer_status({}, is_running=False,
                                 status_lines=['NO_STATUS']) == 'SUBMITTED'

    def test_empty_status_lines_within_grace_is_submitted(self):
        # status_lines == [] is an edge case (the file exists but is
        # empty). Treat it like NO_STATUS: within grace it's still
        # plausibly the script booting; past grace it'd be FAILED.
        mgr = _mgr()
        assert mgr._infer_status(_job(), is_running=False,
                                 status_lines=[]) == 'SUBMITTED'
