"""End-to-end tests for the taxonomic-clustering builder.

Constructs a minimal on-disk project (alignment.fasta + taxonomy.parquet)
behind a lightweight project stub exposing only the methods the builder
touches. No SSH, no network — exercises the full build → cache → manifest
→ reload path.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beak.tui.comparative import (
    active_taxonomic_label,
    build_taxonomic,
    load_active_taxonomic_scores,
    rank_columns,
    rank_summary,
    taxonomic_is_stale,
)


_TARGET = "KGAAA"
# target first; col0 discriminates clade A (K) vs clade B (R).
_ALIGN = [
    ("target", "KGAAA"),
    ("a1", "KGAAA"), ("a2", "KGAAC"), ("a3", "KGACA"),
    ("b1", "RGAAA"), ("b2", "RGAAC"), ("b3", "RGACA"),
]
_TAX = pd.DataFrame(
    {
        "sequence_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
        "phylum": ["Firmicutes"] * 3 + ["Proteobacteria"] * 3,
        "genus": ["Bacillus", "Bacillus", "Staphylococcus",
                  "Escherichia", "Salmonella", "Escherichia"],
        "organism": ["x"] * 6,
    }
)


class FakeProject:
    def __init__(self, root: Path):
        self.path = Path(root)
        self._manifest: dict = {}

    def active_set_name(self) -> str:
        return "default"

    def active_homologs_dir(self, ensure: bool = False) -> Path:
        d = self.path / "homologs" / "sets" / "default"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def homologs_set_dir(self, name: str) -> Path:
        return self.path / "homologs" / "sets" / name

    def target_sequence(self) -> str:
        return _TARGET

    def manifest(self) -> dict:
        return self._manifest

    @contextmanager
    def mutate(self):
        yield self._manifest


@pytest.fixture
def project(tmp_path):
    proj = FakeProject(tmp_path)
    hdir = proj.active_homologs_dir()
    with open(hdir / "alignment.fasta", "w") as fh:
        for sid, seq in _ALIGN:
            fh.write(f">{sid}\n{seq}\n")
    _TAX.to_parquet(hdir / "taxonomy.parquet")
    return proj


def test_rank_columns_lists_qualifying_ranks_coarse_to_fine(project):
    ranks = rank_columns(project)
    assert ranks == ["phylum", "genus"]  # both have >= 2 distinct values


def test_rank_summary_counts_qualifying_clades(project):
    summ = rank_summary(project, "phylum", min_per_clade=3)
    assert summ["n"] == 6
    assert summ["n_clades"] == 2
    assert summ["n_qualifying"] == 2
    assert dict(summ["top"]) == {"Firmicutes": 3, "Proteobacteria": 3}


def test_build_uncertainty_and_cache_and_manifest(project):
    scores = build_taxonomic(project, "phylum", min_per_clade=3)
    assert scores is not None
    assert scores.shape == (len(_TARGET),)
    assert scores[0] > 0.9          # discriminating column
    assert scores[1] == 0.0         # conserved G

    # cache written under the set's taxonomic/ dir
    tdir = project.homologs_set_dir("default") / "taxonomic"
    assert (tdir / "active.npy").exists()
    assert any(f.name.startswith("phylum__") for f in tdir.iterdir())

    tx = project.manifest()["taxonomic"]
    assert tx["active_rank"] == "phylum"
    assert tx["active_score_kind"] == "uncertainty_coefficient"
    assert tx["active_n_clades"] == 2
    assert sorted(tx["active_clades"]) == ["Firmicutes", "Proteobacteria"]
    assert tx["active_set"] == "default"


def test_write_profiles_output_control(project):
    build_taxonomic(project, "phylum", min_per_clade=3, write_profiles=True)
    tdir = project.homologs_set_dir("default") / "taxonomic"
    prof_path = tdir / "profiles__phylum.parquet"
    assert prof_path.exists()
    prof = pd.read_parquet(prof_path)
    assert set(prof.columns) == {"clade", "column", "aa", "freq"}
    # both clades present, 5 alignment columns x 20 AAs each
    assert set(prof["clade"]) == {"Firmicutes", "Proteobacteria"}
    assert prof["column"].nunique() == len(_TARGET)
    assert prof["aa"].nunique() == 20
    assert project.manifest()["taxonomic"]["active_has_profiles"] is True


def test_no_profiles_by_default(project):
    build_taxonomic(project, "phylum", min_per_clade=3)
    tdir = project.homologs_set_dir("default") / "taxonomic"
    assert not (tdir / "profiles__phylum.parquet").exists()
    assert project.manifest()["taxonomic"]["active_has_profiles"] is False


def test_reload_active_matches_build(project):
    built = build_taxonomic(project, "phylum", min_per_clade=3)
    loaded = load_active_taxonomic_scores(project)
    np.testing.assert_array_equal(built, loaded)


def test_permutation_control_switches_score_kind(project):
    build_taxonomic(project, "phylum", min_per_clade=3, n_permutations=50)
    tx = project.manifest()["taxonomic"]
    assert tx["active_score_kind"] == "permutation_zscore"
    assert tx["active_n_permutations"] == 50


def test_degenerate_floor_returns_none(project):
    # min_per_clade=4 leaves < 2 clades (each phylum has 3)
    assert build_taxonomic(project, "phylum", min_per_clade=4) is None


def test_stale_when_active_set_differs(project):
    build_taxonomic(project, "phylum", min_per_clade=3)
    assert taxonomic_is_stale(project) is False
    project.manifest()["taxonomic"]["active_set"] = "other_set"
    assert taxonomic_is_stale(project) is True
    # stale cache refuses to load under the current set
    assert load_active_taxonomic_scores(project) is None


def test_active_label(project):
    build_taxonomic(project, "phylum", min_per_clade=3)
    assert active_taxonomic_label(project) == "phylum · 2 clades · U"
