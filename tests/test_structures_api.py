"""Tests for the PDB/AlphaFold structure discovery and download module."""

import pytest
import pandas as pd
from beak.api.structures import (
    _make_filename,
    _select_structures,
    _structure_score,
    _parse_sifts_response,
)


# ── Sample data ────────────────────────────────────────────────

def _make_row(**kwargs):
    """Helper to create a row dict with defaults."""
    defaults = {
        'uniprot_id': 'P12345',
        'source': 'pdb',
        'structure_id': '6VXX',
        'chain_id': 'A',
        'resolution': 2.8,
        'method': 'X-ray diffraction',
        'coverage_start': 1,
        'coverage_end': 500,
        'download_url': 'https://files.rcsb.org/download/6vxx.cif',
    }
    defaults.update(kwargs)
    return defaults


SAMPLE_SIFTS_RESPONSE = {
    "P0DTC2": {
        "PDB": {
            "6vxx": [
                {
                    "chain_id": "A",
                    "struct_asym_id": "A",
                    "unp_start": 14,
                    "unp_end": 1211,
                    "resolution": 2.8,
                    "experimental_method": "Electron Microscopy",
                }
            ],
            "7dwz": [
                {
                    "chain_id": "A",
                    "struct_asym_id": "A",
                    "unp_start": 319,
                    "unp_end": 541,
                    "resolution": 2.5,
                    "experimental_method": "X-ray diffraction",
                }
            ],
        }
    }
}


# ── Tests ──────────────────────────────────────────────────────

class TestMakeFilename:
    def test_pdb_with_chain(self):
        row = _make_row(source='pdb', uniprot_id='P12345',
                        structure_id='6VXX', chain_id='A')
        assert _make_filename(row) == 'P12345_6VXX_A.cif'

    def test_pdb_no_chain(self):
        row = _make_row(source='pdb', structure_id='6VXX', chain_id='-')
        assert _make_filename(row) == 'P12345_6VXX.cif'

    def test_alphafold(self):
        row = _make_row(source='alphafold', uniprot_id='Q67890',
                        structure_id='AF-Q67890-F1', chain_id='-')
        assert _make_filename(row) == 'Q67890_AF.cif'


class TestStructureScore:
    def test_xray_beats_cryoem(self):
        xray = _make_row(method='X-ray diffraction', resolution=2.0)
        cryoem = _make_row(method='Electron Microscopy', resolution=2.0)
        assert _structure_score(xray) < _structure_score(cryoem)

    def test_lower_resolution_better(self):
        good = _make_row(resolution=1.5)
        bad = _make_row(resolution=3.0)
        assert _structure_score(good) < _structure_score(bad)

    def test_none_resolution_sorts_last(self):
        known = _make_row(resolution=5.0)
        unknown = _make_row(resolution=None)
        assert _structure_score(known) < _structure_score(unknown)

    def test_wider_coverage_better(self):
        wide = _make_row(coverage_start=1, coverage_end=500)
        narrow = _make_row(coverage_start=100, coverage_end=200)
        # Same method and resolution
        assert _structure_score(wide) < _structure_score(narrow)


class TestSelectStructures:
    @pytest.fixture
    def multi_df(self):
        rows = [
            _make_row(uniprot_id='P1', source='pdb',
                      structure_id='AAA', resolution=3.0),
            _make_row(uniprot_id='P1', source='pdb',
                      structure_id='BBB', resolution=1.5),
            _make_row(uniprot_id='P1', source='alphafold',
                      structure_id='AF-P1', resolution=None,
                      method='AlphaFold prediction'),
            _make_row(uniprot_id='P2', source='pdb',
                      structure_id='CCC', resolution=2.0),
            _make_row(uniprot_id='P2', source='alphafold',
                      structure_id='AF-P2', resolution=None,
                      method='AlphaFold prediction'),
        ]
        return pd.DataFrame(rows)

    def test_all_returns_everything(self, multi_df):
        result = _select_structures(multi_df, "all")
        assert len(result) == len(multi_df)

    def test_best_one_per_source_per_id(self, multi_df):
        result = _select_structures(multi_df, "best")
        # P1: best pdb (BBB 1.5A) + alphafold = 2
        # P2: best pdb (CCC) + alphafold = 2
        assert len(result) == 4

    def test_best_picks_lowest_resolution(self, multi_df):
        result = _select_structures(multi_df, "best")
        p1_pdb = result[
            (result['uniprot_id'] == 'P1') & (result['source'] == 'pdb')
        ]
        assert p1_pdb.iloc[0]['structure_id'] == 'BBB'

    def test_top_n(self, multi_df):
        result = _select_structures(multi_df, 2)
        # Top 2 per uniprot_id
        p1 = result[result['uniprot_id'] == 'P1']
        assert len(p1) == 2
        p2 = result[result['uniprot_id'] == 'P2']
        assert len(p2) == 2


class TestParseSiftsResponse:
    def test_parses_sample(self):
        rows = _parse_sifts_response(SAMPLE_SIFTS_RESPONSE)
        assert len(rows) == 2

        # Check first entry
        ids = {r['structure_id'] for r in rows}
        assert '6VXX' in ids
        assert '7DWZ' in ids

    def test_fields_present(self):
        rows = _parse_sifts_response(SAMPLE_SIFTS_RESPONSE)
        for row in rows:
            assert 'uniprot_id' in row
            assert 'source' in row
            assert 'structure_id' in row
            assert 'chain_id' in row
            assert 'resolution' in row
            assert 'download_url' in row
            assert row['source'] == 'pdb'
            assert row['uniprot_id'] == 'P0DTC2'

    def test_empty_response(self):
        assert _parse_sifts_response({}) == []

    def test_coverage_parsed(self):
        rows = _parse_sifts_response(SAMPLE_SIFTS_RESPONSE)
        em_row = [r for r in rows if r['structure_id'] == '6VXX'][0]
        assert em_row['coverage_start'] == 14
        assert em_row['coverage_end'] == 1211
