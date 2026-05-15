"""Tests for the PDB/AlphaFold structure discovery and download module."""

import urllib.error
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
from beak.api import structures as structures_mod
from beak.api.structures import (
    _make_filename,
    _select_structures,
    _structure_score,
    _parse_sifts_response,
    _unlink_if_exists,
    _download_file,
    HTTP_TIMEOUT,
    DOWNLOAD_TIMEOUT,
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


class TestUnlinkIfExists:
    """_unlink_if_exists is the cleanup helper called by _download_file
    on every failure path. It must never raise — the caller is already
    propagating a network error and shouldn't be derailed by a missing
    file or a follow-up unlink failure."""

    def test_removes_existing_file(self, tmp_path):
        p = tmp_path / "partial.cif"
        p.write_bytes(b"junk")
        _unlink_if_exists(str(p))
        assert not p.exists()

    def test_silent_on_missing_file(self, tmp_path):
        # No file present — must not raise.
        _unlink_if_exists(str(tmp_path / "never_existed.cif"))

    def test_swallows_oserror(self, tmp_path):
        # OSError from the underlying os.unlink (e.g. EBUSY on
        # Windows when antivirus has a handle open) is intentionally
        # absorbed so the network exception that triggered cleanup
        # surfaces to the caller unaltered.
        with patch.object(structures_mod.os, "unlink",
                          side_effect=OSError("permission denied")):
            _unlink_if_exists("/nonexistent/path.cif")  # must not raise


class TestDownloadFileCleansUpPartial:
    """Pre-fix: urlretrieve left a half-written `.cif` on the destination
    path when the connection dropped mid-transfer; the next call to the
    structure loader would then parse that corrupt file as if it were a
    valid model. Each test simulates a different mid-transfer failure
    mode and verifies the partial file is removed."""

    def _make_urlopen_that_fails_mid_stream(
        self, dest_path, partial_bytes=b"some partial header\n",
        exc=urllib.error.URLError("connection reset"),
    ):
        """Build an urlopen replacement that writes `partial_bytes` to
        dest_path (mimicking shutil.copyfileobj making progress) and
        then raises during the next read."""
        def fake_urlopen(*_args, **_kwargs):
            # shutil.copyfileobj reads in chunks; first read returns
            # partial bytes (which the real copyfileobj will write to
            # disk via dest's `out.write(...)`), second read raises.
            mock_resp = MagicMock()
            mock_resp.read.side_effect = [partial_bytes, exc]
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = lambda self, *a: None
            return mock_resp
        return fake_urlopen

    def test_partial_file_removed_on_urlerror(self, tmp_path, monkeypatch):
        dest = tmp_path / "AF-Q123.cif"
        # Pre-create the partial file to simulate the shutil.copyfileobj
        # write that lands before the connection drop; the patched
        # urlopen will then raise to trigger the cleanup branch.
        dest.write_bytes(b"header from a partial download\n")

        def boom(*_a, **_kw):
            raise urllib.error.URLError("connection reset")

        monkeypatch.setattr(structures_mod.urllib.request, "urlopen", boom)
        monkeypatch.setattr(structures_mod.time, "sleep", lambda *_a: None)

        with pytest.raises(urllib.error.URLError):
            _download_file("https://example.com/x.cif", str(dest))

        assert not dest.exists(), (
            "Partial file must be removed so the next load can't pick "
            "up corrupt bytes."
        )

    def test_partial_file_removed_on_http_error(self, tmp_path, monkeypatch):
        dest = tmp_path / "AF-Q456.cif"
        dest.write_bytes(b"partial bytes from a prior attempt\n")

        # A non-404, non-429 HTTPError mid-transfer — represents the
        # rare case where the server returns a 500 partway through
        # streaming the response body. Real-world: AlphaFold mirrors
        # have done this when their disk cache rotates.
        def boom(*_a, **_kw):
            raise urllib.error.HTTPError(
                "https://example.com/x.cif", 500,
                "internal server error", {}, None,
            )

        monkeypatch.setattr(structures_mod.urllib.request, "urlopen", boom)
        monkeypatch.setattr(structures_mod.time, "sleep", lambda *_a: None)

        with pytest.raises(urllib.error.HTTPError):
            _download_file("https://example.com/x.cif", str(dest))

        assert not dest.exists()

    def test_404_propagates_as_file_not_found(self, tmp_path, monkeypatch):
        # Regression guard for the existing 404 → FileNotFoundError
        # translation; the cleanup path must not swallow it.
        dest = tmp_path / "AF-missing.cif"

        def boom(*_a, **_kw):
            raise urllib.error.HTTPError(
                "https://example.com/missing.cif", 404,
                "not found", {}, None,
            )

        monkeypatch.setattr(structures_mod.urllib.request, "urlopen", boom)
        monkeypatch.setattr(structures_mod.time, "sleep", lambda *_a: None)

        with pytest.raises(FileNotFoundError):
            _download_file("https://example.com/missing.cif", str(dest))


class TestTimeoutConstants:
    """Smoke-check that the per-request socket timeouts shipped with the
    network-hang fix are still wired up. If a future refactor drops them,
    the calling worker can hang indefinitely on a stalled endpoint."""

    def test_http_timeout_is_positive_int(self):
        assert isinstance(HTTP_TIMEOUT, int) and HTTP_TIMEOUT > 0

    def test_download_timeout_is_positive_int(self):
        assert isinstance(DOWNLOAD_TIMEOUT, int) and DOWNLOAD_TIMEOUT > 0

    def test_download_timeout_is_passed_to_urlopen(
        self, tmp_path, monkeypatch,
    ):
        """The streaming download path must pass DOWNLOAD_TIMEOUT to
        urlopen — otherwise a hung server wedges the worker forever."""
        captured = {}

        def spy(url, *args, **kwargs):
            captured['timeout'] = kwargs.get('timeout')
            # Raise to short-circuit; we only care about the kwargs.
            raise urllib.error.URLError("done capturing")

        monkeypatch.setattr(structures_mod.urllib.request, "urlopen", spy)
        monkeypatch.setattr(structures_mod.time, "sleep", lambda *_a: None)

        with pytest.raises(urllib.error.URLError):
            _download_file("https://example.com/x.cif",
                           str(tmp_path / "x.cif"))
        assert captured['timeout'] == DOWNLOAD_TIMEOUT
