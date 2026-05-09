"""Tests for the PDB-over-AlphaFold cache-lookup priority."""

import pytest

from beak.tui.structure import _cached_structure, cached_structure_path


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# placeholder cif\ndata_test\n")


class TestCachedStructure:
    def test_returns_none_when_directory_missing(self, tmp_path):
        # No `structures/` directory exists at all — caller should
        # short-circuit on None instead of crashing on a stat() call.
        assert _cached_structure("P12345", tmp_path / "missing") is None
        assert cached_structure_path("P12345", tmp_path / "missing") is None

    def test_returns_none_when_directory_empty(self, tmp_path):
        struct_dir = tmp_path / "structures"
        struct_dir.mkdir()
        assert _cached_structure("P12345", struct_dir) is None

    def test_finds_alphafold_cache(self, tmp_path):
        struct_dir = tmp_path / "structures"
        af_path = struct_dir / "P12345_AF.cif"
        _touch(af_path)
        result = _cached_structure("P12345", struct_dir)
        assert result is not None
        path, label = result
        assert path == af_path
        assert label == "AlphaFold"

    def test_finds_pdb_cache(self, tmp_path):
        struct_dir = tmp_path / "structures"
        pdb_path = struct_dir / "P12345_2acp_A.cif"
        _touch(pdb_path)
        result = _cached_structure("P12345", struct_dir)
        assert result is not None
        path, label = result
        assert path == pdb_path
        assert label == "PDB 2acp_A"

    def test_pdb_preferred_over_alphafold(self, tmp_path):
        # Both caches present — PDB wins so users get the experimental
        # structure they downloaded over an auto-fetched AF prediction.
        struct_dir = tmp_path / "structures"
        _touch(struct_dir / "P12345_AF.cif")
        pdb_path = struct_dir / "P12345_2acp_A.cif"
        _touch(pdb_path)
        result = _cached_structure("P12345", struct_dir)
        assert result is not None
        path, label = result
        assert path == pdb_path
        assert label.startswith("PDB ")

    def test_doesnt_match_other_uniprot_ids(self, tmp_path):
        # `P12345_AF.cif` should not match a query for `P123` — the
        # glob is anchored on the full id + underscore. Otherwise a
        # short id could spuriously inherit another protein's cache.
        struct_dir = tmp_path / "structures"
        _touch(struct_dir / "P12345_AF.cif")
        assert _cached_structure("P123", struct_dir) is None

    def test_path_helper_returns_just_path(self, tmp_path):
        struct_dir = tmp_path / "structures"
        af_path = struct_dir / "P12345_AF.cif"
        _touch(af_path)
        assert cached_structure_path("P12345", struct_dir) == af_path


class TestBfactorPalette:
    """Physical-consistency guards: the bfactor mode is its own thing,
    not pLDDT under a different name. Tests pin down the palette
    range, axis title, and color gradient direction so a regression
    that re-aliases bfactor → pLDDT (or swaps the gradient) is caught
    here."""

    def test_bfactor_palette_registered(self):
        from beak.viz.chimerax import _PALETTES, _AXIS_TITLES
        assert "bfactor" in _PALETTES
        palette, score_range, sentinel = _PALETTES["bfactor"]
        assert score_range == (0.0, 50.0)
        assert _AXIS_TITLES["bfactor"] == "B-factor (Å²)"
        # Palette starts at low B = blue, ends at high B = red.
        assert palette.startswith("0,blue")
        assert palette.endswith("50,red")

    def test_bfactor_color_thermal_direction(self):
        # Low B → blue-ish, high B → red-ish. The mid (25 Å²) is
        # white. This is the inverse of pLDDT, where high values
        # are blue (high confidence) — distinct gradients for
        # distinct physical quantities.
        from beak.tui.structure import bfactor_color, plddt_color
        low_b = bfactor_color(5.0)
        high_b = bfactor_color(45.0)
        # Low B has more blue than red.
        low_r = int(low_b[1:3], 16)
        low_b_chan = int(low_b[5:7], 16)
        assert low_b_chan > low_r
        # High B has more red than blue.
        high_r = int(high_b[1:3], 16)
        high_b_chan = int(high_b[5:7], 16)
        assert high_r > high_b_chan
        # Mid is white-ish.
        mid = bfactor_color(25.0)
        assert mid == "#FFFFFF"
        # Distinct from pLDDT — value 95 in pLDDT mode is blue
        # (high confidence), but in bfactor mode 95 is saturated red.
        assert plddt_color(95.0) != bfactor_color(95.0)

    def test_color_for_mode_dispatches_bfactor(self):
        from beak.tui.structure import color_for_mode, bfactor_color
        assert color_for_mode(20.0, mode="bfactor") == bfactor_color(20.0)


class TestEffectiveMode:
    """`StructureView._effective_mode` swaps `plddt` → `bfactor` when
    a PDB structure is loaded. Verified by patching the source label
    on a freshly-instantiated StructureView (the resolver depends on
    no other widget state, so a heavy fixture isn't needed)."""

    def test_resolves_plddt_to_bfactor_for_pdb(self):
        # We can't easily stand up a full StructureView in a unit test
        # (it expects a Textual app), so reach into the resolver
        # directly. The logic is a one-liner method but drives both
        # the canvas rendering and the export title.
        from beak.tui.widgets.structure_view import StructureView

        # Bypass __init__ — we only need the two attributes the
        # resolver reads.
        sv = StructureView.__new__(StructureView)
        sv._color_mode = "plddt"
        sv._source_label = "PDB 2acp_A"
        assert sv._effective_mode() == "bfactor"

    def test_keeps_plddt_for_alphafold(self):
        from beak.tui.widgets.structure_view import StructureView
        sv = StructureView.__new__(StructureView)
        sv._color_mode = "plddt"
        sv._source_label = "AlphaFold"
        assert sv._effective_mode() == "plddt"

    def test_other_modes_passthrough_regardless_of_source(self):
        # Conservation / SASA / differential all have a single
        # physical interpretation, so the source label doesn't matter
        # for those — they should always render as themselves.
        from beak.tui.widgets.structure_view import StructureView
        sv = StructureView.__new__(StructureView)
        for mode in ("conservation", "sasa", "differential", "pfam"):
            for source in ("AlphaFold", "PDB 1xyz_A"):
                sv._color_mode = mode
                sv._source_label = source
                assert sv._effective_mode() == mode


class TestProjectTargetToCif:
    """`project_target_to_cif` is the bridge that lets a PDB structure
    with partial UniProt coverage render conservation / SASA / etc.
    from the target-indexed source arrays. AF cifs collapse to an
    identity copy; PDB cifs trim to the residues they actually have."""

    def test_alphafold_case_is_identity(self):
        # AF cif sequence == target sequence → projection returns the
        # same values (possibly trimmed if cif is shorter).
        import numpy as np
        from beak.tui.structure import project_target_to_cif

        target = "MKVAYIAKQ"
        scalar = np.arange(len(target), dtype=np.float32)
        out = project_target_to_cif(scalar, target, target)
        assert np.array_equal(out, scalar)

    def test_partial_coverage_trims_to_cif(self):
        # PDB covers residues 2-5 of a 6-aa target. The projection
        # should pick up scalar[1..4] and discard the residues outside
        # the cif's coverage (target positions 1 and 6).
        import numpy as np
        from beak.tui.structure import project_target_to_cif

        target = "MKVAYI"          # positions 1..6
        cif = "KVAY"                # covers positions 2..5
        scalar = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
        out = project_target_to_cif(scalar, target, cif)
        assert out.tolist() == [20, 30, 40, 50]

    def test_unmapped_cif_residues_get_sentinel(self):
        # A cif residue that doesn't align to the target (e.g. an
        # extra residue at the C-terminus) should land on the
        # sentinel value rather than something arbitrary.
        import numpy as np
        from beak.tui.structure import project_target_to_cif

        target = "MKVAY"            # 5 aa
        cif = "KVAYZZZ"              # last 3 are unaligned
        scalar = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        out = project_target_to_cif(scalar, target, cif, sentinel=-1.0)
        # First 4 cif residues align to target positions 2..5.
        assert out[:4].tolist() == [20, 30, 40, 50]
        # The trailing 3 cif residues have no target alignment.
        assert all(v == -1.0 for v in out[4:])

    def test_empty_inputs(self):
        import numpy as np
        from beak.tui.structure import project_target_to_cif
        assert len(project_target_to_cif(np.array([]), "M", "")) == 0
        assert len(project_target_to_cif(None, "M", "M")) == 1


class TestThreeToOne:
    """Quick sanity check on the residue code map — surfaces a missing
    entry in `_THREE_TO_ONE` that would otherwise corrupt projections
    by sprinkling 'X' through the cif sequence."""

    def test_canonical_residues(self):
        from beak.tui.structure import _three_to_one
        assert _three_to_one("ALA") == "A"
        assert _three_to_one("TRP") == "W"
        assert _three_to_one("VAL") == "V"

    def test_lowercase_input(self):
        from beak.tui.structure import _three_to_one
        assert _three_to_one("ala") == "A"

    def test_non_canonical_falls_back_to_x(self):
        from beak.tui.structure import _three_to_one
        # Modified residues we don't track explicitly become 'X' so
        # they show up in pairwise alignment as a mismatch rather than
        # truncating the sequence.
        assert _three_to_one("XXX") == "X"

    def test_selenomethionine(self):
        # MSE (selenomethionine) is common in crystal structures and
        # should map to 'M' for alignment purposes.
        from beak.tui.structure import _three_to_one
        assert _three_to_one("MSE") == "M"
