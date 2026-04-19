"""Tests for HmmerScan domtblout parsing."""

import pytest
from beak.remote.hmmer import HmmerScan


# Sample domtblout output (simplified from real hmmscan output)
SAMPLE_DOMTBLOUT = """\
#                                                                            --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord
# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target
#--- -------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------
Pkinase              PF00069.29   260 MY_KINASE            -            742   2.1e-89  298.5   0.3   1   1   1.3e-92   2.1e-89  298.5   0.3     1   260   492   735   492   735 0.97 Protein kinase domain
Recep_L_domain       PF01030.24   150 MY_KINASE            -            742   3.4e-22   78.2   0.1   1   1   2.1e-25   3.4e-22   78.2   0.1     3   148    52   168    50   170 0.91 Receptor L domain
fn3                  PF00041.22    84 MY_KINASE            -            742   1.7e-08   33.5   0.0   1   1   1.1e-11   1.7e-08   33.5   0.0     2    83   303   381   302   382 0.88 Fibronectin type III domain
"""

EMPTY_DOMTBLOUT = ""

COMMENTS_ONLY = """\
#                                                                            --- full sequence ---
# target name        accession   tlen query name
#--- -------------- ---------- -----
"""


@pytest.fixture
def parser():
    """Create an HmmerScan instance for testing parse method.

    We can't test scan() without SSH, but _parse_domtblout is pure logic.
    """
    # HmmerScan.__init__ requires a Connection, but we only test parsing
    # so we create an instance with a dummy connection via __new__
    instance = object.__new__(HmmerScan)
    return instance


def test_parse_domtblout(parser):
    """Test parsing a typical domtblout with multiple domains."""
    hits = parser._parse_domtblout(SAMPLE_DOMTBLOUT)

    assert len(hits) == 3

    # Should be sorted by i_evalue
    assert hits[0]['pfam_id'] == 'PF00069'
    assert hits[1]['pfam_id'] == 'PF01030'
    assert hits[2]['pfam_id'] == 'PF00041'

    # Check first hit fields
    kinase = hits[0]
    assert kinase['pfam_name'] == 'Pkinase'
    assert kinase['description'] == 'Protein kinase domain'
    assert kinase['evalue'] == pytest.approx(2.1e-89)
    assert kinase['score'] == pytest.approx(298.5)
    assert kinase['bias'] == pytest.approx(0.3)
    assert kinase['c_evalue'] == pytest.approx(1.3e-92)
    assert kinase['i_evalue'] == pytest.approx(2.1e-89)
    assert kinase['hmm_from'] == 1
    assert kinase['hmm_to'] == 260
    assert kinase['ali_from'] == 492
    assert kinase['ali_to'] == 735
    assert kinase['env_from'] == 492
    assert kinase['env_to'] == 735


def test_parse_domtblout_version_stripping(parser):
    """Pfam accession version suffixes should be stripped."""
    hits = parser._parse_domtblout(SAMPLE_DOMTBLOUT)

    # PF00069.29 -> PF00069
    assert hits[0]['pfam_id'] == 'PF00069'
    assert '.' not in hits[0]['pfam_id']

    # PF01030.24 -> PF01030
    assert hits[1]['pfam_id'] == 'PF01030'


def test_parse_domtblout_empty(parser):
    """Empty input should return empty list."""
    assert parser._parse_domtblout(EMPTY_DOMTBLOUT) == []


def test_parse_domtblout_comments_only(parser):
    """Input with only comment lines should return empty list."""
    assert parser._parse_domtblout(COMMENTS_ONLY) == []


def test_parse_domtblout_sorted_by_evalue(parser):
    """Results should be sorted by i_evalue ascending."""
    hits = parser._parse_domtblout(SAMPLE_DOMTBLOUT)

    evalues = [h['i_evalue'] for h in hits]
    assert evalues == sorted(evalues)
