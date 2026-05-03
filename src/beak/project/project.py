"""BeakProject: target-centric analysis hub.

Projects live under ~/.beak/projects/<name>/. v0 only handles target/
initialization, manifest I/O, and disk-size accounting. add-homologs,
add-structures, import, refresh, etc. land in subsequent commits.
"""

import json
import re
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .manifest import read_manifest, write_manifest

PROJECTS_DIR = Path.home() / ".beak" / "projects"

_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$')

# Cached size in [stats] is reused for this many seconds in cached_size().
_SIZE_CACHE_TTL = 3600

# Per-path RLocks shared across BeakProject instances pointing at the
# same directory. Threads can recurse (e.g. update_active_set calling
# write inside the lock), so RLock not Lock.
_MANIFEST_LOCKS: Dict[str, threading.RLock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _MANIFEST_LOCKS_GUARD:
        lk = _MANIFEST_LOCKS.get(key)
        if lk is None:
            lk = _MANIFEST_LOCKS[key] = threading.RLock()
        return lk


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """In-place recursive dict merge — used by `update_*_set` helpers
    so passing `remote={"job_id": x}` doesn't blow away sibling keys
    like `remote.search_database` that other callers wrote earlier."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


_MODEL_SLUG_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def _model_slug(model: str) -> str:
    """Sanitize a model name for use as a filesystem path component.

    HF-style identifiers like "EvolutionaryScale/esmc-300m-2024-12"
    contain `/`; the slug replaces any non-`[A-Za-z0-9_.-]` character
    with `_`. Intentionally not lossy in the common case
    (`esm2_t33_650M_UR50D` round-trips to itself).
    """
    return _MODEL_SLUG_RE.sub("_", model)


class BeakProjectError(Exception):
    pass


class BeakProject:
    """Handle to a project on disk. Cheap to construct — does not read I/O."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name

    @classmethod
    def init(
        cls,
        name: str,
        uniprot_id: Optional[str] = None,
        sequence_file: Optional[str] = None,
        description: str = "",
    ) -> "BeakProject":
        if not _NAME_RE.match(name):
            raise BeakProjectError(
                f"Invalid project name '{name}'. Use letters, digits, '_' or '-' "
                "(must start with letter/digit, max 64 chars)."
            )
        if uniprot_id and sequence_file:
            raise BeakProjectError("Pass either --uniprot or --sequence, not both.")
        if not uniprot_id and not sequence_file:
            raise BeakProjectError("One of --uniprot or --sequence is required.")

        path = PROJECTS_DIR / name
        if path.exists():
            raise BeakProjectError(f"Project '{name}' already exists at {path}")

        target_dir = path / "target"
        target_dir.mkdir(parents=True)

        try:
            if uniprot_id:
                meta = cls._populate_target_from_uniprot(target_dir, uniprot_id)
            else:
                meta = cls._populate_target_from_file(target_dir, Path(sequence_file))
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

        manifest = {
            "project": {
                "name": name,
                "created_at": datetime.now(),
                "description": description,
            },
            "target": meta,
        }
        write_manifest(path / "beak.project.toml", manifest)
        return cls(path)

    @classmethod
    def load(cls, name: str) -> "BeakProject":
        path = PROJECTS_DIR / name
        if not (path / "beak.project.toml").exists():
            raise BeakProjectError(f"Project '{name}' not found at {path}")
        return cls(path)

    def rename(self, new_name: str) -> None:
        """Rename the project: move the directory and update the manifest.

        Raises BeakProjectError on invalid name, name collision, or any
        FS failure during the move (in which case the original layout is
        preserved).
        """
        if new_name == self.name:
            return
        if not _NAME_RE.match(new_name):
            raise BeakProjectError(
                f"Invalid project name '{new_name}'. Use letters, digits, "
                "'_' or '-' (must start with letter/digit, max 64 chars)."
            )
        new_path = PROJECTS_DIR / new_name
        if new_path.exists():
            raise BeakProjectError(
                f"A project named '{new_name}' already exists."
            )
        try:
            self.path.rename(new_path)
        except OSError as e:
            raise BeakProjectError(f"Could not rename directory: {e}") from e
        self.path = new_path
        self.name = new_name
        try:
            with self.mutate() as m:
                m.setdefault("project", {})["name"] = new_name
        except Exception:  # noqa: BLE001 — best-effort manifest sync
            pass

    @classmethod
    def list_projects(cls) -> List["BeakProject"]:
        if not PROJECTS_DIR.exists():
            return []
        return [
            cls(p) for p in sorted(PROJECTS_DIR.iterdir())
            if p.is_dir() and (p / "beak.project.toml").exists()
        ]

    @property
    def manifest_path(self) -> Path:
        return self.path / "beak.project.toml"

    @property
    def target_sequence_path(self) -> Path:
        return self.path / "target" / "sequence.fasta"

    def manifest(self) -> Dict[str, Any]:
        return read_manifest(self.manifest_path)

    def write(self, data: Dict[str, Any]) -> None:
        # Always serialize writes against the per-project lock so two
        # workers reading-modifying-writing the manifest in parallel
        # can't clobber each other.
        with _lock_for(self.path):
            write_manifest(self.manifest_path, data)

    @contextmanager
    def mutate(self):
        """Atomic read-modify-write of the manifest.

        Use this from any worker that needs to mutate the manifest:

            with project.mutate() as m:
                m.setdefault("embeddings", {})["n_embeddings"] = n

        The manifest is read inside the lock and written back when the
        block exits, so a concurrent `_pull_homologs_now` can't read a
        stale manifest, miss your update, and clobber it on its own
        write. Cheap when uncontended (a single non-recursive lock
        acquisition); a few microseconds when contended.
        """
        with _lock_for(self.path):
            data = read_manifest(self.manifest_path)
            yield data
            write_manifest(self.manifest_path, data)

    # ---- multi-set homologs ----
    #
    # Each project can host several sets of hits/alignments under
    # `homologs/sets/<name>/`. The manifest's `[homologs]` section
    # carries `active = <name>` and a list of `[[homologs.sets]]`
    # entries with their own counts + remote job IDs. Old projects
    # (one set, files at `homologs/<file>`) are migrated transparently
    # the first time `active_homologs_dir()` is called.

    DEFAULT_SET_NAME = "default"

    def active_set_name(self) -> str:
        """Name of the active homolog set, falling back gracefully.

        If `homologs.active` points at a name that no longer exists in
        `homologs.sets` (e.g. after a delete that removed the active
        set without picking a successor), prefer the first set that
        does exist over returning a stale name. If there are no sets
        at all, fall back to `DEFAULT_SET_NAME` — this is the literal
        sentinel callers like `active_homologs_dir` use to compute the
        path of a not-yet-created set, not a claim that a "default" set
        is real.
        """
        m = self.manifest()
        homologs = m.get("homologs") or {}
        sets = homologs.get("sets") or []
        names = [s.get("name") for s in sets if s.get("name")]
        active = homologs.get("active")
        if active and active in names:
            return active
        if names:
            return names[0]
        return self.DEFAULT_SET_NAME

    def active_homologs_dir(self, ensure: bool = False) -> Path:
        """Path to the active set's homologs directory.

        Migrates old single-set projects on first call. Pass
        `ensure=True` to mkdir -p the result.
        """
        self._migrate_homologs_to_sets()
        d = self.path / "homologs" / "sets" / self.active_set_name()
        if ensure:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def homologs_set_dir(self, name: str) -> Path:
        """Path to a named set's directory (no I/O; doesn't have to exist)."""
        self._migrate_homologs_to_sets()
        return self.path / "homologs" / "sets" / name

    def delete_homolog_set(self, name: str) -> bool:
        """Remove a set from the manifest and rmtree its directory.

        If the active set was deleted, falls back to the first remaining
        set (or clears `[homologs]` entirely if none remain). Returns
        True on success, False if `name` was unknown.
        """
        import shutil
        self._migrate_homologs_to_sets()
        with self.mutate() as m:
            homologs = m.get("homologs") or {}
            sets = list(homologs.get("sets") or [])
            new_sets = [s for s in sets if s.get("name") != name]
            if len(new_sets) == len(sets):
                return False

            d = self.homologs_set_dir(name)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

            if new_sets:
                homologs["sets"] = new_sets
                if homologs.get("active") == name:
                    homologs["active"] = new_sets[0].get("name", self.DEFAULT_SET_NAME)
                m["homologs"] = homologs
            else:
                m.pop("homologs", None)
        return True

    def rename_homolog_set(self, old_name: str, new_name: str) -> bool:
        """Rename a homolog set: move its directory + update the manifest.

        Validates the new name against the same regex as project names so
        downstream paths stay safe. If the active set is being renamed,
        the manifest's `active` pointer is updated to match. Raises
        BeakProjectError on invalid name, collision, or unknown source;
        returns False only when the source set isn't in the manifest.
        """
        if not _NAME_RE.match(new_name):
            raise BeakProjectError(
                f"Invalid set name '{new_name}'. Use letters, digits, "
                "'_' or '-' (must start with letter/digit, max 64 chars)."
            )
        if old_name == new_name:
            return True

        self._migrate_homologs_to_sets()
        sets = self.homologs_sets()
        if not any(s.get("name") == old_name for s in sets):
            return False
        if any(s.get("name") == new_name for s in sets):
            raise BeakProjectError(
                f"A set named '{new_name}' already exists."
            )

        # Move the on-disk directory first; any failure surfaces before
        # we touch the manifest, leaving state consistent.
        old_dir = self.homologs_set_dir(old_name)
        new_dir = self.homologs_set_dir(new_name)
        if old_dir.exists():
            try:
                old_dir.rename(new_dir)
            except OSError as e:
                raise BeakProjectError(
                    f"Could not move set directory: {e}"
                ) from e

        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            for s in homologs.get("sets") or []:
                if s.get("name") == old_name:
                    s["name"] = new_name
                    break
            if homologs.get("active") == old_name:
                homologs["active"] = new_name
        return True

    def filter_homolog_set_by_length(
        self, set_name: str, min_len: int, max_len: int
    ) -> int:
        """Filter a set's `sequences.fasta` to a length range, in place.

        Stream-rewrites the FASTA via a temp file + atomic rename so a
        crash mid-write doesn't leave a half-written hit list. Clears
        the alignment, conservation cache, and the manifest's
        `n_aligned` / `remote.align_job_id` for this set, since the
        old alignment no longer matches the on-disk hits.

        Raises BeakProjectError when the set has no FASTA, when min/max
        are invalid, or when the filter rejects every sequence.
        Returns the number of sequences kept.
        """
        if min_len < 0 or max_len < 0 or min_len > max_len:
            raise BeakProjectError(
                f"Invalid length range [{min_len}, {max_len}]."
            )

        self._migrate_homologs_to_sets()
        set_dir = self.homologs_set_dir(set_name)
        fasta = set_dir / "sequences.fasta"
        if not fasta.exists():
            raise BeakProjectError(
                f"No sequences.fasta in set '{set_name}'."
            )

        tmp = fasta.with_suffix(".filtered.tmp")
        n_kept = 0
        n_total = 0

        try:
            with open(fasta) as src, open(tmp, "w") as dst:
                header: Optional[str] = None
                seq_parts: List[str] = []

                def _flush() -> None:
                    nonlocal n_kept, n_total
                    if header is None:
                        return
                    n_total += 1
                    seq = "".join(seq_parts)
                    if min_len <= len(seq) <= max_len:
                        dst.write(header)
                        dst.write(seq + "\n")
                        n_kept += 1

                for line in src:
                    if line.startswith(">"):
                        _flush()
                        header = line
                        seq_parts = []
                    else:
                        seq_parts.append(line.strip())
                _flush()
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise BeakProjectError(f"Could not rewrite FASTA: {e}") from e

        if n_kept == 0:
            tmp.unlink(missing_ok=True)
            raise BeakProjectError(
                f"No sequences in [{min_len}, {max_len}] aa "
                f"(checked {n_total:,})."
            )

        tmp.replace(fasta)

        # Clear stale alignment artifacts — the old alignment was built
        # over a superset of these sequences, so the column map and
        # conservation scores no longer correspond to the FASTA on disk.
        # The parsed-alignment cache sidecar is dropped explicitly so a
        # stale ~10 MB file doesn't sit around orphaned.
        from ..alignments.cache import invalidate_cache
        for name in ("alignment.fasta", "conservation.npy"):
            f = set_dir / name
            if f.exists():
                if name == "alignment.fasta":
                    invalidate_cache(f)
                f.unlink()

        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            for s in homologs.get("sets") or []:
                if s.get("name") == set_name:
                    s["n_homologs"] = n_kept
                    s.pop("n_aligned", None)
                    remote = s.get("remote") or {}
                    remote.pop("align_job_id", None)
                    s["remote"] = remote
                    s["last_updated"] = datetime.now()
                    break

        return n_kept

    def dedupe_homolog_set(self, set_name: str) -> tuple[int, int]:
        """Drop exact-duplicate sequences from a set's `sequences.fasta`.

        Stream-rewrites the FASTA via temp file + atomic rename, keeping
        the first occurrence of each unique sequence (case-insensitive,
        whitespace-stripped). Clears the alignment, conservation cache,
        alignment-cache sidecar, and the manifest's `n_aligned` /
        `remote.align_job_id` for this set, since the old alignment was
        built over the redundant FASTA.

        Returns ``(n_kept, n_dropped)``. Raises BeakProjectError when
        the set has no FASTA. When nothing is duplicated, the file is
        left untouched and ``n_dropped == 0`` — no alignment is cleared
        in that case.
        """
        self._migrate_homologs_to_sets()
        set_dir = self.homologs_set_dir(set_name)
        fasta = set_dir / "sequences.fasta"
        if not fasta.exists():
            raise BeakProjectError(
                f"No sequences.fasta in set '{set_name}'."
            )

        tmp = fasta.with_suffix(".dedup.tmp")
        n_kept = 0
        n_dropped = 0
        seen: set[str] = set()

        try:
            with open(fasta) as src, open(tmp, "w") as dst:
                header: Optional[str] = None
                seq_parts: List[str] = []

                def _flush() -> None:
                    nonlocal n_kept, n_dropped
                    if header is None:
                        return
                    seq = "".join(seq_parts)
                    key = seq.upper().strip()
                    if not key:
                        # Empty / whitespace-only record — skip without
                        # counting toward dedup. Same conservative posture
                        # as the search-pipeline filter that strips them
                        # before alignment.
                        return
                    if key in seen:
                        n_dropped += 1
                        return
                    seen.add(key)
                    dst.write(header)
                    dst.write(seq + "\n")
                    n_kept += 1

                for line in src:
                    if line.startswith(">"):
                        _flush()
                        header = line
                        seq_parts = []
                    else:
                        seq_parts.append(line.strip())
                _flush()
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise BeakProjectError(
                f"Could not rewrite FASTA: {e}"
            ) from e

        if n_dropped == 0:
            # Nothing changed — leave the FASTA + alignment intact.
            tmp.unlink(missing_ok=True)
            return n_kept, 0

        tmp.replace(fasta)

        # Clear stale alignment artifacts — the old alignment was built
        # over the redundant FASTA, so column counts no longer match.
        from ..alignments.cache import invalidate_cache
        for name in ("alignment.fasta", "conservation.npy"):
            f = set_dir / name
            if f.exists():
                if name == "alignment.fasta":
                    invalidate_cache(f)
                f.unlink()

        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            for s in homologs.get("sets") or []:
                if s.get("name") == set_name:
                    s["n_homologs"] = n_kept
                    s.pop("n_aligned", None)
                    remote = s.get("remote") or {}
                    remote.pop("align_job_id", None)
                    s["remote"] = remote
                    s["last_updated"] = datetime.now()
                    break

        return n_kept, n_dropped

    def homologs_sets(self) -> List[Dict[str, Any]]:
        """Return the list of homolog sets in the manifest (after migration)."""
        self._migrate_homologs_to_sets()
        m = self.manifest()
        return list((m.get("homologs") or {}).get("sets") or [])

    def active_set(self) -> Optional[Dict[str, Any]]:
        """Return the active set's dict, or None if no sets exist yet."""
        active = self.active_set_name()
        for s in self.homologs_sets():
            if s.get("name") == active:
                return s
        return None

    def update_active_set(self, **fields) -> None:
        """Merge `fields` into the active set's dict; create it if absent."""
        self._migrate_homologs_to_sets()
        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            active = homologs.setdefault("active", self.DEFAULT_SET_NAME)
            sets = homologs.setdefault("sets", [])
            for s in sets:
                if s.get("name") == active:
                    s.update(fields)
                    return
            sets.append({"name": active, **fields})

    def set_active_set(self, name: str) -> bool:
        """Switch the active set. Returns True if `name` exists."""
        self._migrate_homologs_to_sets()
        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            for s in homologs.get("sets") or []:
                if s.get("name") == name:
                    homologs["active"] = name
                    return True
            return False

    def add_homolog_set(self, name: str, **fields) -> None:
        """Register a new set in the manifest. Doesn't create files."""
        if not _NAME_RE.match(name):
            raise BeakProjectError(
                f"Invalid set name '{name}'. Use letters, digits, '_' or '-'."
            )
        self._migrate_homologs_to_sets()
        with self.mutate() as m:
            homologs = m.setdefault("homologs", {})
            sets = homologs.setdefault("sets", [])
            for s in sets:
                if s.get("name") == name:
                    # Already exists — update in place.
                    s.update(fields)
                    return
            sets.append({"name": name, **fields})
            if "active" not in homologs:
                homologs["active"] = name

    # ---- multi-model, multi-set embeddings ----
    #
    # Embeddings are keyed by the *pair* (homolog set, model) so a
    # single set can be embedded with several models in parallel and
    # everything coexists. The manifest's `embeddings.sets` is a flat
    # list of entries each containing both `source_homologs_set` and
    # `model`; uniqueness is enforced on the pair.
    #
    # Files live under `embeddings/<set_name>/<model_slug>/...` so
    # different models of the same set don't clobber each other.
    # Legacy projects (one flat dir, no model subdir) migrate on first
    # access — `_migrate_embeddings_to_sets` handles both the
    # pre-multi-set and pre-multi-model layouts.

    def active_embeddings_dir(
        self, ensure: bool = False, model: Optional[str] = None,
    ) -> Path:
        """Path to the active homolog set's embeddings directory.

        Pass `model=` for the per-model subdir; omit to get the
        legacy flat per-set dir (used during migration / fallback).
        """
        return self.embeddings_set_dir(
            self.active_set_name(), model=model, ensure=ensure,
        )

    def embeddings_set_dir(
        self,
        set_name: str,
        model: Optional[str] = None,
        ensure: bool = False,
    ) -> Path:
        """Path to a named (set, model) embeddings dir.

        Without `model`, returns the per-set parent — useful for
        listing all models or resetting an entire set's embeddings.
        """
        self._migrate_embeddings_to_sets()
        d = self.path / "embeddings" / set_name
        if model is not None:
            d = d / _model_slug(model)
        if ensure:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def embeddings_sets(self) -> List[Dict[str, Any]]:
        """All embeddings entries, after migration. May be empty."""
        self._migrate_embeddings_to_sets()
        m = self.manifest()
        return list((m.get("embeddings") or {}).get("sets") or [])

    def embeddings_models_for_set(self, set_name: str) -> List[Dict[str, Any]]:
        """Every embeddings entry whose `source_homologs_set` matches.

        Order is deterministic: most-recently-updated first, so
        callers that take `[0]` get the freshest model for that set.
        """
        from datetime import datetime, timezone

        def _ts(entry: Dict[str, Any]) -> float:
            v = entry.get("last_updated")
            if isinstance(v, datetime):
                ts = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
                return ts.timestamp()
            try:
                return datetime.fromisoformat(str(v)).timestamp()
            except (ValueError, TypeError):
                return 0.0

        entries = [
            s for s in self.embeddings_sets()
            if s.get("source_homologs_set") == set_name
        ]
        entries.sort(key=_ts, reverse=True)
        return entries

    def active_embeddings_set(
        self, model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Embeddings entry for (active set, `model`).

        With `model=None`, returns the most-recently-updated entry for
        the active set, mirroring the previous single-model behavior.
        """
        active = self.active_set_name()
        candidates = self.embeddings_models_for_set(active)
        if not candidates:
            return None
        if model is None:
            return candidates[0]
        for s in candidates:
            if s.get("model") == model:
                return s
        return None

    def update_active_embeddings_set(self, model: str, **fields) -> None:
        """Merge `fields` into the (active set, `model`) entry; create
        if absent. `model` is required so the entry is uniquely keyed
        on the (set, model) pair."""
        self._migrate_embeddings_to_sets()
        if not model:
            raise BeakProjectError(
                "update_active_embeddings_set requires a `model` name "
                "so the (set, model) pair is uniquely identifiable."
            )
        active = self.active_set_name()
        with self.mutate() as m:
            emb = m.setdefault("embeddings", {})
            sets = emb.setdefault("sets", [])
            for s in sets:
                if (s.get("source_homologs_set") == active
                        and s.get("model") == model):
                    _deep_update(s, fields)
                    return
            sets.append({
                "source_homologs_set": active,
                "model": model,
                **fields,
            })

    def update_embeddings_set_by_job(self, job_id: str, **fields) -> bool:
        """Find the embeddings entry holding `job_id` and merge `fields`."""
        self._migrate_embeddings_to_sets()
        with self.mutate() as m:
            sets = (m.get("embeddings") or {}).get("sets") or []
            for s in sets:
                if (s.get("remote") or {}).get("job_id") == job_id:
                    _deep_update(s, fields)
                    return True
        return False

    def embeddings_dir_for_job(self, job_id: str) -> Optional[Path]:
        """Where to extract a given job's tarball — the (set, model)
        directory of the entry that owns this job_id."""
        for s in self.embeddings_sets():
            if (s.get("remote") or {}).get("job_id") == job_id:
                return self.embeddings_set_dir(
                    s["source_homologs_set"], model=s.get("model"),
                )
        return None

    def delete_embeddings_set(
        self, set_name: str, model: Optional[str] = None,
    ) -> None:
        """Drop embeddings entries for `(set_name, model)`.

        With `model=None`, removes ALL models for the set (the
        original bulk-reset semantics). With a specific `model`,
        removes just that one entry and leaves siblings intact —
        which is what a row-targeted "Remove" needs so the user can
        clear a single failed model without nuking the working ones.
        """
        import shutil
        self._migrate_embeddings_to_sets()
        if model is None:
            d = self.embeddings_set_dir(set_name)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        else:
            d = self.embeddings_set_dir(set_name, model=model)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        with self.mutate() as m:
            emb = m.get("embeddings") or {}
            sets = emb.get("sets") or []
            if model is None:
                new_sets = [
                    s for s in sets
                    if s.get("source_homologs_set") != set_name
                ]
            else:
                new_sets = [
                    s for s in sets
                    if not (
                        s.get("source_homologs_set") == set_name
                        and s.get("model") == model
                    )
                ]
            if new_sets:
                emb["sets"] = new_sets
                m["embeddings"] = emb
            else:
                m.pop("embeddings", None)

    def delete_active_embeddings_set(
        self, model: Optional[str] = None,
    ) -> None:
        """Convenience wrapper — `delete_embeddings_set` on the active set."""
        self.delete_embeddings_set(self.active_set_name(), model=model)

    def _migrate_embeddings_to_sets(self) -> None:
        """Two-stage migration: flat → per-set, and per-set → per-(set,model).

        Stage 1 covers the very-early shape with one flat `[embeddings]`
        block (no `sets` key at all). Stage 2 covers the per-set shape
        introduced before multi-model support — each set has chunks
        directly under `embeddings/<set>/` instead of nested under
        `embeddings/<set>/<model_slug>/`.
        """
        m = self.manifest()
        emb = m.get("embeddings")
        if not emb:
            return

        # ---- Stage 1: flat → sets[] ----
        if "sets" not in emb:
            src = (
                emb.get("source_homologs_set")
                or self.active_set_name()
                or self.DEFAULT_SET_NAME
            )
            legacy_keys = (
                "n_embeddings", "model", "remote", "last_updated",
                "source_homologs_set",
            )
            new_entry: Dict[str, Any] = {"source_homologs_set": src}
            for k in legacy_keys:
                if k in emb and k != "source_homologs_set":
                    new_entry[k] = emb[k]
            with self.mutate() as data:
                data["embeddings"] = {"sets": [new_entry]}

            # `embeddings/*` → `embeddings/<src>/*` via a staging
            # rename so we don't try to mkdir the target inside the
            # source we're about to move.
            emb_root = self.path / "embeddings"
            if emb_root.exists() and any(emb_root.iterdir()):
                target = emb_root / src
                if not target.exists():
                    staging = self.path / f".embeddings_migrate_{src}"
                    if staging.exists():
                        import shutil
                        shutil.rmtree(staging, ignore_errors=True)
                    emb_root.rename(staging)
                    emb_root.mkdir()
                    staging.rename(target)

        # ---- Stage 2: per-set → per-(set, model) ----
        # If any entry already has a model recorded but the on-disk
        # files still live at `embeddings/<set>/chunks/...` (i.e. no
        # `<model_slug>/` segment), nest them under the right slug.
        m = self.manifest()
        emb = m.get("embeddings") or {}
        sets = emb.get("sets") or []
        for entry in sets:
            set_name = entry.get("source_homologs_set")
            model = entry.get("model")
            if not set_name or not model:
                continue
            set_dir = self.path / "embeddings" / set_name
            if not set_dir.exists():
                continue
            # Heuristic: legacy layout has `chunks/` directly under
            # the set dir; the new layout has `<model_slug>/chunks/`.
            legacy_chunks = set_dir / "chunks"
            if not legacy_chunks.exists():
                continue
            slug = _model_slug(model)
            target = set_dir / slug
            if target.exists():
                continue  # already migrated by another path
            staging = set_dir.parent / f".emb_model_migrate_{set_name}_{slug}"
            if staging.exists():
                import shutil
                shutil.rmtree(staging, ignore_errors=True)
            # Move every child of set_dir into staging, then rename
            # staging to set_dir/<slug>. This preserves files at the
            # set_dir root that aren't `chunks/` (rare) by also
            # nesting them under <slug>, which is the right behavior:
            # everything under set_dir was that one model's output.
            staging.mkdir(parents=True)
            for child in list(set_dir.iterdir()):
                child.rename(staging / child.name)
            staging.rename(target)

    def _migrate_homologs_to_sets(self) -> None:
        """One-shot: convert legacy `homologs/{file}` + flat manifest
        keys to the new `homologs/sets/default/{file}` layout."""
        m = self.manifest()
        homologs = m.get("homologs")
        if not homologs:
            return
        if "sets" in homologs:
            return  # already migrated

        # Pull old top-level fields into a virtual default set.
        legacy_keys = (
            "n_homologs", "n_aligned", "source", "remote", "last_updated",
        )
        if not any(homologs.get(k) for k in legacy_keys):
            # Nothing to migrate yet (e.g. brand-new project).
            return

        default_set: Dict[str, Any] = {"name": self.DEFAULT_SET_NAME}
        for k in legacy_keys:
            if k in homologs:
                default_set[k] = homologs.pop(k)

        # Move files: homologs/X → homologs/sets/default/X
        old_dir = self.path / "homologs"
        new_dir = old_dir / "sets" / self.DEFAULT_SET_NAME
        if old_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            for entry in list(old_dir.iterdir()):
                if entry.is_file():
                    entry.rename(new_dir / entry.name)

        homologs["active"] = self.DEFAULT_SET_NAME
        homologs["sets"] = [default_set]
        m["homologs"] = homologs
        self.write(m)

    def target_sequence(self) -> str:
        text = self.target_sequence_path.read_text()
        return ''.join(
            ln.strip() for ln in text.splitlines()
            if ln and not ln.startswith('>')
        )

    # ---- size accounting ----

    def disk_usage(self) -> int:
        if not self.path.exists():
            return 0
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def disk_usage_by_layer(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for sub in sorted(self.path.iterdir()):
            if sub.is_dir():
                result[sub.name] = sum(
                    p.stat().st_size for p in sub.rglob("*") if p.is_file()
                )
        return result

    def cached_size(self, ttl_seconds: int = _SIZE_CACHE_TTL) -> int:
        """Return cached size if fresh, else recompute and persist.

        Walking the tree on every `project list` invocation gets slow
        once embeddings land — cache in [stats] with a TTL.
        """
        m = self.manifest()
        stats = m.get("stats") or {}
        last = stats.get("last_computed")
        if last and "size_bytes" in stats:
            try:
                ts = last if isinstance(last, datetime) else datetime.fromisoformat(str(last))
                if (datetime.now() - ts).total_seconds() < ttl_seconds:
                    return int(stats["size_bytes"])
            except (ValueError, TypeError):
                pass
        return int(self.refresh_stats()["size_bytes"])

    def refresh_stats(self) -> Dict[str, Any]:
        size = self.disk_usage()
        m = self.manifest()
        m["stats"] = {
            "size_bytes": size,
            "last_computed": datetime.now(),
        }
        self.write(m)
        return m["stats"]

    def status_summary(self) -> str:
        """Single-word status for list views.

        Surfaces whichever job kind is in flight across the *whole*
        project (any homolog set's search/align, taxonomy, any
        embedding model). Previously this only reported the active
        set's search status, so a project with a running ESM-C job
        on a fully-aligned set rendered as `"ready"` even though
        work was actively happening.

        Order of specificity (returned label): one of
            "new" — no homologs anywhere, no jobs running.
            "search" — a search job is in flight.
            "align" — an alignment job is in flight.
            "tax" — a taxonomy job is in flight.
            "embed" — an embedding job is in flight.
            "running" — multiple jobs in flight.
            "ready" — everything settled.
        """
        m = self.manifest()
        sets = (m.get("homologs") or {}).get("sets") or []

        # Collect job_ids by kind, only counting jobs that haven't yet
        # produced their downstream artifact (no n_homologs / n_aligned
        # / n_assigned / n_embeddings). This filters out stale-but-
        # tagged ids from completed runs.
        in_flight: Dict[str, List[str]] = {
            "search": [], "align": [], "tax": [], "embed": [],
        }

        for s in sets:
            remote = s.get("remote") or {}
            if remote.get("search_job_id") and not s.get("n_homologs"):
                in_flight["search"].append(remote["search_job_id"])
            if (remote.get("align_job_id")
                    and s.get("n_homologs")
                    and not s.get("n_aligned")):
                in_flight["align"].append(remote["align_job_id"])

        tax = m.get("taxonomy") or {}
        tax_jid = (tax.get("remote") or {}).get("job_id")
        if tax_jid and not tax.get("n_assigned"):
            in_flight["tax"].append(tax_jid)

        for e in (m.get("embeddings") or {}).get("sets") or []:
            remote = e.get("remote") or {}
            if remote.get("job_id") and not e.get("n_embeddings"):
                in_flight["embed"].append(remote["job_id"])

        # Confirm against the local jobs.json snapshot — a job that
        # was cancelled / failed but never had its manifest entry
        # cleared shouldn't claim the project's "in flight" badge.
        live_states = {"RUNNING", "SUBMITTED", "QUEUED"}
        db: Dict[str, Any] = {}
        db_path = Path.home() / ".beak" / "jobs.json"
        if db_path.exists():
            try:
                with open(db_path) as f:
                    db = json.load(f)
            except (json.JSONDecodeError, OSError):
                db = {}

        def _is_live(jid: str) -> bool:
            info = db.get(jid)
            if info is None:
                # Unknown to local DB — assume submitted; the layers
                # panel poll worker will catch up shortly.
                return True
            return info.get("status", "RUNNING") in live_states

        live_kinds = [
            k for k, ids in in_flight.items()
            if any(_is_live(j) for j in ids)
        ]

        if len(live_kinds) > 1:
            return "running"
        if live_kinds:
            return live_kinds[0]

        # Nothing in flight. "ready" if any homologs landed anywhere,
        # "new" if the project is still empty.
        has_homologs = any(s.get("n_homologs") for s in sets)
        return "ready" if has_homologs else "new"

    def status(self) -> Dict[str, Any]:
        m = self.manifest()
        layers = {
            "target": bool(m.get("target")),
            "homologs": bool(m.get("homologs")),
            "domains": bool(m.get("domains")),
            "structures": bool(m.get("structures")),
            "experiments": bool(m.get("experiments")),
        }
        return {
            "name": self.name,
            "path": str(self.path),
            "manifest": m,
            "layers": layers,
            "size": self.disk_usage(),
            "size_by_layer": self.disk_usage_by_layer(),
        }

    # ---- target population ----

    @staticmethod
    def _populate_target_from_uniprot(target_dir: Path, accession: str) -> Dict[str, Any]:
        from ..api.uniprot import fetch_uniprot

        fetched = Path(fetch_uniprot(accession, output_dir=str(target_dir)))
        seq_path = target_dir / "sequence.fasta"
        if fetched != seq_path:
            shutil.move(str(fetched), str(seq_path))

        return _parse_target_fasta(seq_path, fallback_accession=accession)

    @staticmethod
    def _populate_target_from_file(target_dir: Path, src: Path) -> Dict[str, Any]:
        if not src.exists():
            raise BeakProjectError(f"Sequence file not found: {src}")

        text = src.read_text()
        if not text.lstrip().startswith('>'):
            raise BeakProjectError(f"Not a FASTA file (no '>' header): {src}")

        seq_path = target_dir / "sequence.fasta"
        seq_path.write_text(text if text.endswith('\n') else text + '\n')

        return _parse_target_fasta(seq_path)


def _parse_target_fasta(seq_path: Path, fallback_accession: Optional[str] = None) -> Dict[str, Any]:
    from ..sequence import parse_uniprot_header

    lines = seq_path.read_text().splitlines()
    header = lines[0].lstrip('>') if lines and lines[0].startswith('>') else ''
    seq = ''.join(ln.strip() for ln in lines[1:] if ln and not ln.startswith('>'))
    if not seq:
        raise BeakProjectError(f"Empty sequence in: {seq_path}")

    parsed = parse_uniprot_header(header)
    description = parsed.get('description', '') or ''
    organism = _extract_uniprot_field(description, 'OS')
    gene = _extract_uniprot_field(description, 'GN')

    meta: Dict[str, Any] = {
        "sequence_file": "target/sequence.fasta",
        "length": len(seq),
    }
    accession = parsed.get('accession') or fallback_accession
    if accession:
        meta["uniprot_id"] = accession
    if parsed.get('name'):
        meta["uniprot_name"] = parsed['name']
    if gene:
        meta["gene_name"] = gene
    if organism:
        meta["organism"] = organism

    (seq_path.parent / "metadata.json").write_text(
        json.dumps({"header": header, **meta}, indent=2)
    )
    return meta


def _extract_uniprot_field(description: str, key: str) -> Optional[str]:
    """Pull an OS=/GN=/etc. value out of a UniProt FASTA description.

    UniProt headers terminate each field at the next ' XX=' tag.
    """
    m = re.search(rf'\b{key}=(.+?)(?=\s+[A-Z]{{2}}=|$)', description)
    return m.group(1).strip() if m else None
