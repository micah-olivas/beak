"""Unit tests for remote/hmmer.py domtblout parsing."""

import pytest

from beak.remote.hmmer import HmmerScan


# A minimal two-hit domtblout fixture. Fields 1..22 are:
#   target name, target acc, tlen, query name, query acc, qlen,
#   full E-value, full score, full bias,
#   dom #, dom total, dom c-Evalue, dom i-Evalue, dom score, dom bias,
#   hmm from, hmm to, ali from, ali to, env from, env to, acc, description
DOMTBLOUT = """#
#                                                     ---------- this line is a comment
# target name        accession   tlen query name       accession  qlen  fullE  fullS bias  dom# total cEval iEval  score  bias  hmmf hmmt  alif  alit envf  envt acc description
Pkinase               PF00069.27   264 query1           -           350 2.1e-60 200.1  0.0    1     1 1e-65 2.3e-64  210.5  0.0    5  260    10   268    5   273 0.98 Protein kinase domain
SH3_1                 PF00018.30    58 query1           -           350 1.5e-10  40.2  0.0    1     1 5e-12 1.1e-11   38.7  0.0    2   58   100   156   98   160 0.95 SH3 domain
"""


def _parser():
    # HmmerScan.__init__ requires a live Connection; bypass it for pure parsing.
    return HmmerScan.__new__(HmmerScan)


class TestParseDomtblout:
    def test_parses_two_hits(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        assert len(hits) == 2

    def test_strips_version_suffix_from_accession(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        assert {h['pfam_id'] for h in hits} == {'PF00069', 'PF00018'}

    def test_keeps_pfam_name(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        assert {h['pfam_name'] for h in hits} == {'Pkinase', 'SH3_1'}

    def test_numeric_fields_are_typed(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        hit = hits[0]
        assert isinstance(hit['i_evalue'], float)
        assert isinstance(hit['score'], float)
        assert isinstance(hit['ali_from'], int)
        assert isinstance(hit['ali_to'], int)

    def test_sorted_by_i_evalue_ascending(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        ievalues = [h['i_evalue'] for h in hits]
        assert ievalues == sorted(ievalues)
        assert hits[0]['pfam_id'] == 'PF00069'  # smaller i_evalue

    def test_description_preserved(self):
        hits = _parser()._parse_domtblout(DOMTBLOUT)
        descriptions = {h['pfam_name']: h['description'] for h in hits}
        assert 'kinase' in descriptions['Pkinase'].lower()
        assert 'SH3' in descriptions['SH3_1']

    def test_blank_input_returns_empty_list(self):
        assert _parser()._parse_domtblout("") == []

    def test_only_comments_returns_empty_list(self):
        text = "# header\n# another comment\n"
        assert _parser()._parse_domtblout(text) == []

    def test_malformed_lines_skipped(self):
        # A short line (< 22 fields) is silently dropped rather than raising
        text = DOMTBLOUT + "\nbroken line too short\n"
        hits = _parser()._parse_domtblout(text)
        assert len(hits) == 2
