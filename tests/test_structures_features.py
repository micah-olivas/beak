"""Tests for the structures feature extraction module."""

import pytest
import pandas as pd

from beak.structures.features import (
    extract_structure_features,
    _parse_chain,
    _extract_plddt,
    _get_representative_atom,
    _compute_contacts_and_wcn,
    _parse_ss_from_mmcif,
)


# ── mmCIF builder ────────────────────────────────────────────────

def _atom(aid, name, comp, chain, eid, seq, x, y, z, bfac):
    return (
        f"ATOM {aid} C {name} . {comp} {chain} {eid} {seq} ? "
        f"{x:.3f} {y:.3f} {z:.3f} 1.00 {bfac:.2f} {seq} {chain} 1"
    )


def _wrap_mmcif(atom_lines, entities, extra=""):
    ent = "\n#\n".join(
        f"_entity.id   {e}\n_entity.type polymer" for e, _ in entities
    )
    poly = "\n#\n".join(
        f"_entity_poly.entity_id  {e}\n"
        f"_entity_poly.type       'polypeptide(L)'\n"
        f"_entity_poly.pdbx_strand_id {c}" for e, c in entities
    )
    atoms = "\n".join(atom_lines)
    return f"""data_test
#
{ent}
#
{poly}
#
{extra}
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
def linear_cif(tmp_path):
    """8 residues along x-axis: 5 ALA + 3 GLY."""
    lines = []
    aid = 1
    for i in range(5):
        x = i * 3.8
        lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, i+1, x, 0, 0, 95-i*5))
        lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, i+1, x+1.5, 0, 0, 95-i*5))
        aid += 2
    for i in range(3):
        x = (5 + i) * 3.8
        lines.append(_atom(aid, 'CA', 'GLY', 'A', 1, 6+i, x, 0, 0, 70-i*5))
        aid += 1
    path = tmp_path / "linear.cif"
    path.write_text(_wrap_mmcif(lines, [(1, 'A')]))
    return str(path)


@pytest.fixture
def cluster_cif(tmp_path):
    """6 residues: 1-5 compact, 6 isolated at 50A."""
    positions = [
        (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.5, 2.6, 0.0),
        (1.5, 5.2, 0.0), (0.0, 7.8, 0.0),
    ]
    lines = []
    aid = 1
    for i, (x, y, z) in enumerate(positions):
        lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, i+1, x, y, z, 95-i*5))
        lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, i+1, x+1, y, z, 95-i*5))
        aid += 2
    lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, 6, 50, 50, 50, 70))
    lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, 6, 51, 50, 50, 70))
    path = tmp_path / "cluster.cif"
    path.write_text(_wrap_mmcif(lines, [(1, 'A')]))
    return str(path)


@pytest.fixture
def ss_cif(tmp_path):
    """6 ALA with helix (1-3) and sheet (5-6) annotations."""
    lines = []
    aid = 1
    for i in range(6):
        x = i * 3.8
        lines.append(_atom(aid, 'CA', 'ALA', 'A', 1, i+1, x, 0, 0, 90-i*5))
        lines.append(_atom(aid+1, 'CB', 'ALA', 'A', 1, i+1, x+1.5, 0, 0, 90-i*5))
        aid += 2
    ss = """\
loop_
_struct_conf.conf_type_id
_struct_conf.beg_auth_asym_id
_struct_conf.beg_auth_seq_id
_struct_conf.end_auth_asym_id
_struct_conf.end_auth_seq_id
HELX_P A 1 A 3
#
loop_
_struct_sheet_range.beg_auth_asym_id
_struct_sheet_range.beg_auth_seq_id
_struct_sheet_range.end_auth_asym_id
_struct_sheet_range.end_auth_seq_id
A 5 A 6
#"""
    path = tmp_path / "ss.cif"
    path.write_text(_wrap_mmcif(lines, [(1, 'A')], extra=ss))
    return str(path)


# ── Tests: extract_structure_features ────────────────────────────

class TestExtractStructureFeatures:
    def test_returns_expected_columns(self, linear_cif):
        df = extract_structure_features(linear_cif, skip_sasa=True)
        expected = {
            'residue_number', 'residue_name', 'plddt',
            'secondary_structure', 'sasa', 'n_contacts', 'burial_wcn',
        }
        assert expected.issubset(set(df.columns))

    def test_one_row_per_residue(self, linear_cif):
        df = extract_structure_features(linear_cif, skip_sasa=True)
        assert len(df) == 8

    def test_residue_numbers_present(self, linear_cif):
        df = extract_structure_features(linear_cif, skip_sasa=True)
        assert list(df['residue_number']) == list(range(1, 9))

    def test_plddt_values_present(self, linear_cif):
        df = extract_structure_features(linear_cif, skip_sasa=True)
        assert df['plddt'].notna().all()
        assert (df['plddt'] > 0).all()

    def test_missing_chain_raises(self, linear_cif):
        with pytest.raises(ValueError, match="Chain 'Z'"):
            extract_structure_features(linear_cif, chain_id='Z')

    def test_invalid_path_raises(self):
        with pytest.raises(Exception):
            extract_structure_features("/nonexistent/file.cif")

    def test_skip_flags(self, linear_cif):
        df = extract_structure_features(linear_cif, skip_dssp=True, skip_sasa=True)
        assert len(df) == 8


# ── Tests: pLDDT ─────────────────────────────────────────────────

class TestPlddt:
    def test_first_higher_than_last(self, linear_cif):
        structure, chain, _ = _parse_chain(linear_cif)
        plddt = _extract_plddt(chain.get_polymer())
        assert plddt[1] > plddt[8]

    def test_all_residues_have_values(self, linear_cif):
        structure, chain, _ = _parse_chain(linear_cif)
        plddt = _extract_plddt(chain.get_polymer())
        assert len(plddt) == 8


# ── Tests: contacts and WCN ──────────────────────────────────────

class TestContactsAndWcn:
    def test_linear_no_distant_contacts(self, linear_cif):
        """Linear chain: CB spacing >> 8A for seq separation >= 4."""
        structure, chain, _ = _parse_chain(linear_cif)
        contacts, _ = _compute_contacts_and_wcn(chain.get_polymer(), structure)
        for c in contacts.values():
            assert c == 0

    def test_cluster_has_contacts(self, cluster_cif):
        structure, chain, _ = _parse_chain(cluster_cif)
        contacts, _ = _compute_contacts_and_wcn(chain.get_polymer(), structure)
        assert any(c > 0 for c in contacts.values())

    def test_wcn_positive(self, linear_cif):
        structure, chain, _ = _parse_chain(linear_cif)
        _, wcn = _compute_contacts_and_wcn(chain.get_polymer(), structure)
        assert all(w > 0 for w in wcn.values())

    def test_isolated_residue_lowest_wcn(self, cluster_cif):
        structure, chain, _ = _parse_chain(cluster_cif)
        _, wcn = _compute_contacts_and_wcn(chain.get_polymer(), structure)
        assert wcn[6] < min(wcn[r] for r in [1, 2, 3, 4, 5])

    def test_glycine_has_wcn(self, linear_cif):
        """GLY (residues 6-8) use CA and still get WCN values."""
        structure, chain, _ = _parse_chain(linear_cif)
        _, wcn = _compute_contacts_and_wcn(chain.get_polymer(), structure)
        for r in [6, 7, 8]:
            assert r in wcn


# ── Tests: representative atom ───────────────────────────────────

class TestRepresentativeAtom:
    def test_ala_uses_cb(self, linear_cif):
        _, chain, _ = _parse_chain(linear_cif)
        atom = _get_representative_atom(chain.get_polymer()[0])
        assert atom.name == 'CB'

    def test_gly_uses_ca(self, linear_cif):
        _, chain, _ = _parse_chain(linear_cif)
        atom = _get_representative_atom(chain.get_polymer()[5])
        assert atom.name == 'CA'


# ── Tests: secondary structure fallback ──────────────────────────

class TestSecondaryStructureFallback:
    def test_helix_parsed(self, ss_cif):
        ss = _parse_ss_from_mmcif(ss_cif, 'A')
        for r in [1, 2, 3]:
            assert ss.get(r) == 'H'

    def test_sheet_parsed(self, ss_cif):
        ss = _parse_ss_from_mmcif(ss_cif, 'A')
        for r in [5, 6]:
            assert ss.get(r) == 'E'

    def test_unassigned_is_coil(self, ss_cif):
        ss = _parse_ss_from_mmcif(ss_cif, 'A')
        assert ss.get(4, 'C') == 'C'

    def test_no_annotations_returns_empty(self, linear_cif):
        ss = _parse_ss_from_mmcif(linear_cif, 'A')
        assert len(ss) == 0
