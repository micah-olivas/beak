"""Tests for atomic JSON writes on the remote-job manager.

The projects index (`~/beak_projects/.index.json`) and per-project
metadata (`<project_dir>/.metadata.json`) are both read on every UI
launch. Pre-fix, both used a bare ``open(..., 'w'); json.dump(...)``
pattern; a crash or kill mid-write left a truncated JSON that broke
the index for *every* job in the file, not just the one being
written. These tests pin the fix in place.

Tests bypass ``RemoteJobManager.__init__`` (which would try to open
an SSH connection) and instead exercise the bound methods against
an instance constructed via ``object.__new__``. The methods only
touch ``self.LOCAL_PROJECTS_DIR``; nothing else from ``__init__`` is
required.
"""

import json
import threading

import pytest

from beak.remote.base import RemoteJobManager


def _bare_manager(local_dir):
    """Build a manager that skips __init__ and only knows about
    LOCAL_PROJECTS_DIR — enough to call the atomic-write code paths."""
    rjm = object.__new__(RemoteJobManager)
    rjm.LOCAL_PROJECTS_DIR = local_dir
    return rjm


class TestUpdateProjectsIndexAtomic:
    def test_no_tmp_sibling_after_success(self, tmp_path):
        """A successful write must leave the `.tmp` staging file gone.
        os.replace renames it, so any leftover means we wrote via the
        old non-atomic path."""
        rjm = _bare_manager(tmp_path)
        rjm._update_projects_index(
            "job1", tmp_path / "p1", "search", "test",
        )
        siblings = {child.name for child in tmp_path.iterdir()}
        assert ".index.json" in siblings
        assert ".index.json.tmp" not in siblings

    def test_payload_round_trips(self, tmp_path):
        rjm = _bare_manager(tmp_path)
        rjm._update_projects_index(
            "job1", tmp_path / "p1", "search", "alpha",
        )
        rjm._update_projects_index(
            "job2", tmp_path / "p2", "align", "beta",
        )
        idx = json.loads((tmp_path / ".index.json").read_text())
        assert idx["job1"]["name"] == "alpha"
        assert idx["job2"]["name"] == "beta"
        assert idx["job1"]["job_type"] == "search"
        assert idx["job2"]["job_type"] == "align"

    def test_concurrent_writers_produce_valid_json(self, tmp_path):
        """Two threads hammering the same index file must always leave
        a well-formed JSON document at the user-visible path. The atomic
        write rules out a *torn* intermediate; concurrent races may still
        clobber each other's entries (no lock, so last-writer-wins) but
        the file is never left mid-replace."""
        rjm = _bare_manager(tmp_path)
        barrier = threading.Barrier(2)
        errors = []

        def worker(prefix):
            barrier.wait()
            try:
                for i in range(40):
                    rjm._update_projects_index(
                        f"{prefix}-{i}",
                        tmp_path / f"{prefix}-{i}",
                        "search", f"{prefix}-{i}",
                    )
            except Exception as e:  # pragma: no cover - lock failures shouldn't occur
                errors.append(e)

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == []
        # File must be parseable JSON regardless of which writer won.
        data = json.loads((tmp_path / ".index.json").read_text())
        assert isinstance(data, dict)
        # No `.tmp` leftover even under contention.
        assert not (tmp_path / ".index.json.tmp").exists()


class TestCreateProjectMetadataAtomic:
    def test_no_tmp_sibling_after_success(self, tmp_path, monkeypatch):
        """`create_project` writes `.metadata.json` inside the new
        project dir. After a successful return, the staging tempfile
        must not be visible to the project loader."""
        # Suppress the `_update_projects_index` call's side effects so
        # the test focuses on metadata atomicity; it has its own tests.
        rjm = _bare_manager(tmp_path)
        monkeypatch.setattr(rjm, "_update_projects_index",
                            lambda *a, **kw: None)

        proj = rjm.create_project("jobX", "search", name="my proj")
        meta_files = {child.name for child in proj.iterdir()}
        assert ".metadata.json" in meta_files
        assert ".metadata.json.tmp" not in meta_files

    def test_metadata_payload_round_trips(self, tmp_path, monkeypatch):
        rjm = _bare_manager(tmp_path)
        monkeypatch.setattr(rjm, "_update_projects_index",
                            lambda *a, **kw: None)
        proj = rjm.create_project("jobY", "align", name="round trip")
        meta = json.loads((proj / ".metadata.json").read_text())
        assert meta["job_id"] == "jobY"
        assert meta["job_type"] == "align"
        # `create_project` slugifies the name into the project dir but
        # the metadata's `name` field preserves the original.
        assert meta["name"] == "round trip"


class TestAtomicWritePartialFailure:
    """Simulate a failure between writing the tempfile and the final
    `os.replace`. The user-visible path must remain untouched (or its
    prior content unchanged) — never a torn write."""

    def test_index_unchanged_if_fsync_replaces_fail(
        self, tmp_path, monkeypatch,
    ):
        """If `os.replace` raises (disk full, EXDEV on a tmp-on-another-fs
        setup), the original index — empty in this case — must be
        unchanged and there must be no half-written `.index.json`."""
        rjm = _bare_manager(tmp_path)

        # Pre-populate so we can verify "unchanged" means "same bytes".
        rjm._update_projects_index(
            "j0", tmp_path / "p0", "search", "before",
        )
        original = (tmp_path / ".index.json").read_bytes()

        import beak.remote.base as base_mod

        def boom(*_a, **_kw):
            raise OSError("simulated disk full")

        monkeypatch.setattr(base_mod.os, "replace", boom)

        with pytest.raises(OSError):
            rjm._update_projects_index(
                "j1", tmp_path / "p1", "search", "after",
            )

        # User-visible file unchanged.
        assert (tmp_path / ".index.json").read_bytes() == original
