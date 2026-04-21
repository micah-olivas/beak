"""Tests for the structures coordinate mapping module."""

import pytest
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from beak.structures.mapping import (
    map_alignment_to_target,
    map_target_to_structure,
    _extract_sequence_from_chain,
    align_sequences,
)


# ── mmCIF builder ────────────────────────────────────────────────

def _atom(aid, name, comp, chain, eid, seq, x, y, z, bfac):
    return (
        f"ATOM {aid} C {name} . {comp} {chain} {eid} {seq} ? "
        f"{x:.3f} {y:.3f} {z:.3f} 1.00 {bfac:.2f} {seq} {chain} 1"
    )


def _wrap_mmcif(atom_lines, entities):
    """Build valid mmCIF from atom lines and (entity_id, chain_id) pairs."""
    if len(entities) == 1:
        e, c = entities[0]
        ent = f"_entity.id   {e}\n_entity.type polymer"
        poly = (f"_entity_poly.entity_id  {e}\n"
                f"_entity_poly.type       'polypeptide(L)'\n"
                f"_entity_poly.pdbx_strand_id {c}")
    else:
        ent_rows = "\n".join(f"{e} polymer" for e, _ in entities)
        ent = f"loop_\n_entity.id\n_entity.type\n{ent_rows}"
        poly_rows = "\n".join(f"{e} 'polypeptide(L)' {c}" for e, c in entities)
        poly = (f"loop_\n_entity_poly.entity_id\n_entity_poly.type\n"
                f"_entity_poly.pdbx_strand_id\n{poly_rows}")
    atoms = "\n".join(atom_lines)
    return f"""data_test
#
{ent}
#
{poly}
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
{atoms}
#
"""


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def mini_cif(tmp_path):
    """5 ALA along x-axis."""
    lines = []
    aid = 1
    for i in range(5):
        x = i * 3.8
        lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, i+1, x, 0, 0, 95-i*5))
        lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, i+1, x+1.5, 0, 0, 95-i*5))
        aid += 2
    path = tmp_path / "mini.cif"
    path.write_text(_wrap_mmcif(lines, [(1, 'A')]))
    return str(path)


@pytest.fixture
def two_chain_cif(tmp_path):
    """Chain A: 3 ALA, Chain B: 2 GLY."""
    lines = []
    aid = 1
    for i in range(3):
        x = i * 3.8
        lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, i+1, x, 0, 0, 90-i*5))
        lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, i+1, x+1.5, 0, 0, 90-i*5))
        aid += 2
    for i in range(2):
        x = i * 3.8
        lines.append(_atom(aid, 'CA', 'GLY', 'B', 2, i+1, x, 10, 0, 70-i*5))
        aid += 1
    path = tmp_path / "two_chain.cif"
    path.write_text(_wrap_mmcif(lines, [(1, 'A'), (2, 'B')]))
    return str(path)


def _make_alignment(seqs):
    records = [
        SeqRecord(Seq(seq), id=name, description="")
        for name, seq in seqs
    ]
    return MultipleSeqAlignment(records)


# ── Tests: map_alignment_to_target ───────────────────────────────

class TestMapAlignmentToTarget:
    def test_ungapped_target(self):
        aln = _make_alignment([("target", "ACDEF"), ("other", "AC-EF")])
        result = map_alignment_to_target(aln, "target")
        assert list(result['aln_column']) == [0, 1, 2, 3, 4]
        assert list(result['target_pos']) == [1, 2, 3, 4, 5]

    def test_gapped_target(self):
        aln = _make_alignment([("target", "AC--F"), ("other", "ACDEF")])
        result = map_alignment_to_target(aln, "target")
        assert list(result['aln_column']) == [0, 1, 4]
        assert list(result['target_pos']) == [1, 2, 3]

    def test_target_not_found_raises(self):
        aln = _make_alignment([("seq1", "ACDEF"), ("seq2", "ACDEF")])
        with pytest.raises(KeyError, match="not_here"):
            map_alignment_to_target(aln, "not_here")

    def test_gap_only_columns_excluded(self):
        aln = _make_alignment([("target", "-A-"), ("other", "AAA")])
        result = map_alignment_to_target(aln, "target")
        assert len(result) == 1
        assert result.iloc[0]['aln_column'] == 1

    def test_output_columns(self):
        aln = _make_alignment([("target", "AC"), ("other", "AC")])
        result = map_alignment_to_target(aln, "target")
        assert list(result.columns) == ['aln_column', 'target_pos']


# ── Tests: map_target_to_structure ───────────────────────────────

class TestMapTargetToStructure:
    def test_perfect_match(self, mini_cif):
        result = map_target_to_structure("AAAAA", mini_cif)
        assert len(result) == 5
        assert list(result['target_pos']) == [1, 2, 3, 4, 5]

    def test_target_longer_than_structure(self, mini_cif):
        result = map_target_to_structure("AAAAAAA", mini_cif)
        assert len(result) == 5

    def test_target_shorter_than_structure(self, mini_cif):
        result = map_target_to_structure("AAA", mini_cif)
        assert len(result) == 3

    def test_output_columns(self, mini_cif):
        result = map_target_to_structure("AAAAA", mini_cif)
        assert list(result.columns) == ['target_pos', 'pdb_resnum', 'pdb_resname']

    def test_target_pos_is_one_based(self, mini_cif):
        result = map_target_to_structure("AAAAA", mini_cif)
        assert result.iloc[0]['target_pos'] == 1

    def test_chain_id_selection(self, two_chain_cif):
        result = map_target_to_structure("GG", two_chain_cif, chain_id='B')
        assert len(result) == 2

    def test_missing_chain_raises(self, mini_cif):
        with pytest.raises(ValueError, match="Chain 'Z'"):
            map_target_to_structure("AAA", mini_cif, chain_id='Z')

    def test_default_chain_is_first_polymer(self, two_chain_cif):
        result = map_target_to_structure("AAA", two_chain_cif)
        assert len(result) == 3


# ── Tests: helpers ───────────────────────────────────────────────

class TestExtractSequence:
    def test_returns_correct_sequence(self, mini_cif):
        seq, nums, _ = _extract_sequence_from_chain(mini_cif)
        assert seq == "AAAAA"
        assert nums == [1, 2, 3, 4, 5]

    def test_chain_b_glycines(self, two_chain_cif):
        seq, nums, _ = _extract_sequence_from_chain(two_chain_cif, chain_id='B')
        assert seq == "GG"
        assert nums == [1, 2]


class TestAlignSequences:
    def test_identical(self):
        a, b = align_sequences("ACDEF", "ACDEF")
        assert a == "ACDEF"
        assert b == "ACDEF"

    def test_with_gaps(self):
        a, b = align_sequences("ACDEF", "AEF")
        assert '-' in a or '-' in b
        assert len(a) == len(b)


# ── Tests: composition ───────────────────────────────────────────

class TestComposition:
    def test_alignment_to_structure_pipeline(self, mini_cif):
        aln = _make_alignment([("target", "A-AAA-A"), ("other", "AAAAAAA")])
        aln_map = map_alignment_to_target(aln, "target")
        struct_map = map_target_to_structure("AAAAA", mini_cif)
        merged = aln_map.merge(struct_map, on='target_pos')
        assert 'aln_column' in merged.columns
        assert 'pdb_resnum' in merged.columns
        assert len(merged) == 5
