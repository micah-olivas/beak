"""Tests for atomic + cross-process safe manifest writes.

Locks in the regression-pass fixes for `beak.project.manifest.write_manifest`:
- The user-visible path never observes a torn write — `os.replace` is
  the only mutation against the final filename.
- Concurrent writers (multiple TUI / CLI instances on the same
  project) are serialized by an flock'd sibling lockfile, so the
  second writer can't silently overwrite the first via a races
  temp-name collision.

Each test corresponds to a specific bug the fix addresses. The happy
path round-trip is exercised transitively by both scenarios.
"""

import threading

from beak.project.manifest import read_manifest, write_manifest


def test_write_uses_atomic_replace(tmp_path):
    """The atomic-rename should leave no `.tmp` sibling behind, even
    after multiple writes to the same path. Pre-fix, a crash mid-
    `tomli_w.dump` could leave a partial TOML at the user-visible
    path; this test would have caught that by observing tempfile
    cruft after a successful return."""
    p = tmp_path / "beak.project.toml"
    write_manifest(p, {"a": 1})
    write_manifest(p, {"a": 2})
    siblings = sorted(child.name for child in tmp_path.iterdir())
    assert "beak.project.toml.tmp" not in siblings
    assert read_manifest(p) == {"a": 2}


def test_concurrent_writes_do_not_lose_either_call(tmp_path):
    """Two threads calling write_manifest simultaneously must serialize
    through the flock — neither write should produce a torn file or
    vanish silently. Last-writer-wins is fine; *silently lost mid-
    write* is not.

    Pre-fix had no lock: two threads could both write to
    `beak.project.toml.tmp` and the second `os.replace` could land
    a torn intermediate as the user-visible file.
    """
    p = tmp_path / "beak.project.toml"
    payload_a = {"writer": "a", "value": "x" * 4096}
    payload_b = {"writer": "b", "value": "y" * 4096}

    barrier = threading.Barrier(2)
    errors = []

    def writer(payload):
        barrier.wait()  # maximize the race window
        try:
            for _ in range(20):
                write_manifest(p, payload)
        except Exception as e:  # pragma: no cover - lock failures should not occur
            errors.append(e)

    t1 = threading.Thread(target=writer, args=(payload_a,))
    t2 = threading.Thread(target=writer, args=(payload_b,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert errors == []
    final = read_manifest(p)
    # Whichever thread won the final flock, the file is well-formed and
    # an exact match for one of the two payloads — no torn merge.
    assert final in (payload_a, payload_b)
