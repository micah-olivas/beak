"""Tests for beak.embeddings.load_hit_taxonomy + the shared parse_lineage_df.

The remote convertalis call is integration-level (needs SSH + mmseqs)
and not covered here. These tests exercise the pure parsing / loading
path that both the remote search manager and the client loader share.
"""

import pandas as pd
import pytest

from beak.embeddings import load_hit_taxonomy
from beak.remote.taxonomy import parse_lineage_df, _parse_lineage_string


def _write_tsv(tmp_path, rows):
    """Write a 4-column TSV (target, taxid, organism, lineage) like the
    output of `mmseqs convertalis --format-output target,taxid,taxname,taxlineage`.
    """
    p = tmp_path / "hits_taxonomy.tsv"
    with open(p, 'w') as f:
        for row in rows:
            f.write('\t'.join(str(x) for x in row) + '\n')
    return p


class TestParseLineageString:
    def test_bacterial_full_lineage(self):
        t = _parse_lineage_string(
            '-_root;d_Bacteria;p_Proteobacteria;c_Gammaproteobacteria;'
            'o_Enterobacterales;f_Enterobacteriaceae;g_Escherichia;'
            's_Escherichia coli'
        )
        assert t['domain'] == 'Bacteria'
        assert t['phylum'] == 'Proteobacteria'
        assert t['class'] == 'Gammaproteobacteria'
        assert t['order'] == 'Enterobacterales'
        assert t['family'] == 'Enterobacteriaceae'
        assert t['genus'] == 'Escherichia'
        assert t['species'] == 'Escherichia coli'

    def test_archaeal_lineage(self):
        t = _parse_lineage_string(
            '-_root;d_Archaea;p_Euryarchaeota;c_Methanococci'
        )
        assert t['domain'] == 'Archaea'

    def test_eukaryotic_with_kingdom(self):
        t = _parse_lineage_string('-_root;d_Eukaryota;k_Metazoa;p_Chordata')
        assert t['domain'] == 'Eukaryota'
        assert t['kingdom'] == 'Metazoa'

    def test_empty(self):
        assert _parse_lineage_string('') == {}

    def test_nan(self):
        assert _parse_lineage_string(float('nan')) == {}

    def test_keyword_fallback_without_d_prefix(self):
        # Some lineages come through as NCBI's raw semicolon format
        # without the rank-code prefixes. We still want the domain.
        t = _parse_lineage_string('cellular organisms; Bacteria; Terrabacteria')
        assert t['domain'] == 'Bacteria'

    def test_ranks_with_blank_names_are_skipped(self):
        t = _parse_lineage_string('d_Bacteria;p_;c_Gammaproteobacteria')
        assert t['domain'] == 'Bacteria'
        assert 'phylum' not in t
        assert t['class'] == 'Gammaproteobacteria'


class TestParseLineageDf:
    def test_raises_on_missing_column(self):
        df = pd.DataFrame({'target': ['x'], 'taxid': [1]})
        with pytest.raises(KeyError, match='lineage'):
            parse_lineage_df(df)

    def test_adds_all_expected_rank_columns(self):
        df = pd.DataFrame({'lineage': ['d_Bacteria;p_Firmicutes']})
        out = parse_lineage_df(df)
        for col in ('domain', 'kingdom', 'phylum', 'class',
                    'order', 'family', 'genus', 'species'):
            assert col in out.columns

    def test_mixed_rows(self):
        df = pd.DataFrame({'lineage': [
            'd_Bacteria;p_Proteobacteria;g_Escherichia',
            'd_Archaea;p_Euryarchaeota;g_Methanococcus',
            '',  # empty should produce all-None ranks
        ]})
        out = parse_lineage_df(df)
        assert list(out['domain']) == ['Bacteria', 'Archaea', None]
        assert list(out['genus']) == ['Escherichia', 'Methanococcus', None]


class TestLoadHitTaxonomy:
    def test_missing_file_returns_empty(self, tmp_path):
        out = load_hit_taxonomy(tmp_path / "does_not_exist.tsv")
        assert out.empty

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.tsv"
        p.write_text("")
        out = load_hit_taxonomy(p)
        assert out.empty

    def test_parses_basic_row(self, tmp_path):
        p = _write_tsv(tmp_path, [
            ('UniRef90_P12345', 562, 'Escherichia coli',
             '-_root;d_Bacteria;p_Proteobacteria;g_Escherichia;s_Escherichia coli'),
        ])
        out = load_hit_taxonomy(p)

        assert list(out.index) == ['UniRef90_P12345']
        row = out.iloc[0]
        assert row['taxid'] == 562
        assert row['organism'] == 'Escherichia coli'
        assert row['domain'] == 'Bacteria'
        assert row['phylum'] == 'Proteobacteria'
        assert row['genus'] == 'Escherichia'
        assert row['species'] == 'Escherichia coli'

    def test_deduplicates_on_target(self, tmp_path):
        # MMseqs2 can emit multiple HSPs for the same hit; dedupe.
        p = _write_tsv(tmp_path, [
            ('UniRef90_P12345', 562, 'Ecoli', 'd_Bacteria'),
            ('UniRef90_P12345', 562, 'Ecoli', 'd_Bacteria'),
            ('UniRef90_Q67890', 100, 'Other',  'd_Archaea'),
        ])
        out = load_hit_taxonomy(p)
        assert len(out) == 2
        assert set(out.index) == {'UniRef90_P12345', 'UniRef90_Q67890'}

    def test_index_aligns_with_embedding_seq_ids(self, tmp_path):
        # If the caller does `load_mean_embeddings(...)`, its index is
        # full UniRef accessions (with the prefix). Our output indexes
        # by the same `target` string the search emits, so .loc[...]
        # lookups work without any translation.
        p = _write_tsv(tmp_path, [
            ('UniRef90_A0A087X769', 1, 'org1', 'd_Bacteria;p_X'),
            ('UniRef90_A0A087Y5T1', 2, 'org2', 'd_Archaea;p_Y'),
        ])
        out = load_hit_taxonomy(p)
        emb_index = pd.Index(
            ['UniRef90_A0A087X769', 'UniRef90_A0A087Y5T1'], name='seq_id',
        )
        aligned = out.loc[emb_index, 'domain']
        assert list(aligned) == ['Bacteria', 'Archaea']
