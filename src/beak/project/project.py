"""BeakProject: target-centric analysis hub.

Projects live under ~/.beak/projects/<name>/. v0 only handles target/
initialization, manifest I/O, and disk-size accounting. add-homologs,
add-structures, import, refresh, etc. land in subsequent commits.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import read_manifest, write_manifest

PROJECTS_DIR = Path.home() / ".beak" / "projects"

_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$')

# Cached size in [stats] is reused for this many seconds in cached_size().
_SIZE_CACHE_TTL = 3600


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
            m = self.manifest()
            m.setdefault("project", {})["name"] = new_name
            self.write(m)
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
        m = self.manifest()
        return (m.get("homologs") or {}).get("active", self.DEFAULT_SET_NAME)

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
        m = self.manifest()
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
        self.write(m)
        return True

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
        m = self.manifest()
        homologs = m.setdefault("homologs", {})
        active = homologs.setdefault("active", self.DEFAULT_SET_NAME)
        sets = homologs.setdefault("sets", [])
        for s in sets:
            if s.get("name") == active:
                s.update(fields)
                self.write(m)
                return
        new_set = {"name": active, **fields}
        sets.append(new_set)
        self.write(m)

    def set_active_set(self, name: str) -> bool:
        """Switch the active set. Returns True if `name` exists."""
        self._migrate_homologs_to_sets()
        m = self.manifest()
        homologs = m.setdefault("homologs", {})
        for s in homologs.get("sets") or []:
            if s.get("name") == name:
                homologs["active"] = name
                self.write(m)
                return True
        return False

    def add_homolog_set(self, name: str, **fields) -> None:
        """Register a new set in the manifest. Doesn't create files."""
        if not _NAME_RE.match(name):
            raise BeakProjectError(
                f"Invalid set name '{name}'. Use letters, digits, '_' or '-'."
            )
        self._migrate_homologs_to_sets()
        m = self.manifest()
        homologs = m.setdefault("homologs", {})
        sets = homologs.setdefault("sets", [])
        for s in sets:
            if s.get("name") == name:
                # Already exists — update in place.
                s.update(fields)
                self.write(m)
                return
        sets.append({"name": name, **fields})
        if "active" not in homologs:
            homologs["active"] = name
        self.write(m)

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
        """Single-word status for list views — looks at the active set."""
        active = self.active_set() or {}
        if active.get("n_homologs"):
            return "ready"
        remote = active.get("remote") or {}
        job_id = remote.get("search_job_id")
        if job_id:
            db_path = Path.home() / ".beak" / "jobs.json"
            if db_path.exists():
                try:
                    with open(db_path) as f:
                        jdb = json.load(f)
                    if job_id in jdb:
                        return jdb[job_id].get("status", "RUNNING").lower()
                except (json.JSONDecodeError, OSError):
                    pass
            return "running"
        return "new"

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
