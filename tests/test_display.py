"""Tests for beak.cli.display Rich rendering."""

from io import StringIO

import pytest
from rich.console import Console, Group

from beak.cli.display import render_status, print_status


def _capture(info: dict) -> str:
    """Render status info to a plain-text string (no ANSI codes)."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=80)
    print_status(info, console=console)
    return buf.getvalue()


class TestRenderStatus:
    def test_returns_group(self):
        info = {"job_id": "abc", "name": "test", "status": "RUNNING"}
        result = render_status(info)
        assert isinstance(result, Group)

    def test_completed_status(self):
        info = {
            "job_id": "abc123",
            "name": "my_search",
            "status": "COMPLETED",
            "job_type": "search",
            "runtime": "0:05:32",
            "stages": [
                {"label": "Counting k-mers", "state": "done"},
                {"label": "Aligning", "state": "done"},
            ],
        }
        output = _capture(info)
        assert "BEAK" in output
        assert "my_search" in output
        assert "COMPLETED" in output
        assert "0:05:32" in output
        assert "\u2713" in output  # checkmark
        assert "beak results abc123" in output

    def test_running_status_with_stages(self):
        info = {
            "job_id": "def456",
            "name": "taxonomy_run",
            "status": "RUNNING",
            "job_type": "taxonomy",
            "runtime": "0:02:10",
            "stages": [
                {"label": "Counting k-mers", "state": "done"},
                {"label": "Aligning", "state": "active"},
                {"label": "Computing LCA", "state": "pending"},
            ],
            "last_log_line": "Processing batch 42...",
        }
        output = _capture(info)
        assert "RUNNING" in output
        assert "\u2713" in output   # done icon
        assert "\u25b6" in output   # active icon
        assert "\u25cb" in output   # pending icon
        assert "Processing batch 42" in output

    def test_failed_status(self):
        info = {
            "job_id": "fail789",
            "name": "broken_job",
            "status": "FAILED",
            "job_type": "align",
            "runtime": "0:00:15",
            "stages": [],
        }
        output = _capture(info)
        assert "FAILED" in output
        assert "beak log fail789" in output

    def test_no_stages(self):
        info = {
            "job_id": "x",
            "name": "simple",
            "status": "SUBMITTED",
            "stages": [],
        }
        output = _capture(info)
        assert "SUBMITTED" in output

    def test_pipeline_stages(self):
        info = {
            "job_id": "pipe1",
            "name": "my_pipeline",
            "status": "RUNNING",
            "job_type": "pipeline",
            "runtime": "0:10:00",
            "stages": [
                {"label": "Step 1: search", "state": "done"},
                {"label": "Step 2: taxonomy", "state": "active"},
                {"label": "Step 3: align", "state": "pending"},
            ],
            "last_log_line": None,
        }
        output = _capture(info)
        assert "Step 1: search" in output
        assert "Step 2: taxonomy" in output
        assert "Step 3: align" in output
