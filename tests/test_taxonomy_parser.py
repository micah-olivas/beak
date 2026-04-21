"""Unit tests for remote/taxonomy.py lineage parsing."""

import pandas as pd
import pytest

from beak.remote.taxonomy import MMseqsTaxonomy


def _parser():
    # MMseqsTaxonomy.__init__ requires a live Connection; bypass it for pure parsing.
    return MMseqsTaxonomy.__new__(MMseqsTaxonomy)


class TestParseTaxonomyLineage:
    def test_returns_df_unchanged_when_lineage_missing(self):
        df = pd.DataFrame({'taxid': [562], 'scientific_name': ['E. coli']})
        result = _parser().parse_taxonomy_lineage(df)
        assert list(result.columns) == ['taxid', 'scientific_name']

    def test_bacterial_lineage_populates_rank_columns(self):
        df = pd.DataFrame({'lineage': [
            '-_cellular organisms;d_Bacteria;p_Proteobacteria;c_Gammaproteobacteria;'
            'o_Enterobacterales;f_Enterobacteriaceae;g_Escherichia;s_Escherichia coli'
        ]})
        result = _parser().parse_taxonomy_lineage(df)

        row = result.iloc[0]
        assert row['domain'] == 'Bacteria'
        assert row['phylum'] == 'Proteobacteria'
        assert row['class'] == 'Gammaproteobacteria'
        assert row['order'] == 'Enterobacterales'
        assert row['family'] == 'Enterobacteriaceae'
        assert row['genus'] == 'Escherichia'
        assert row['species'] == 'Escherichia coli'

    def test_archaeal_lineage_sets_domain(self):
        df = pd.DataFrame({'lineage': [
            '-_cellular organisms;d_Archaea;p_Euryarchaeota;c_Methanococci'
        ]})
        result = _parser().parse_taxonomy_lineage(df)
        assert result.iloc[0]['domain'] == 'Archaea'
        assert result.iloc[0]['phylum'] == 'Euryarchaeota'

    def test_eukaryotic_lineage_sets_domain(self):
        df = pd.DataFrame({'lineage': [
            '-_cellular organisms;d_Eukaryota;k_Metazoa;p_Chordata'
        ]})
        result = _parser().parse_taxonomy_lineage(df)
        assert result.iloc[0]['domain'] == 'Eukaryota'
        assert result.iloc[0]['kingdom'] == 'Metazoa'

    def test_empty_lineage_produces_none_values(self):
        df = pd.DataFrame({'lineage': ['']})
        result = _parser().parse_taxonomy_lineage(df)
        assert pd.isna(result.iloc[0]['domain'])
        assert pd.isna(result.iloc[0]['genus'])

    def test_nan_lineage_produces_none_values(self):
        df = pd.DataFrame({'lineage': [pd.NA]})
        result = _parser().parse_taxonomy_lineage(df)
        assert pd.isna(result.iloc[0]['domain'])

    def test_all_rank_columns_added(self):
        df = pd.DataFrame({'lineage': ['d_Bacteria']})
        result = _parser().parse_taxonomy_lineage(df)
        for rank in ['domain', 'kingdom', 'phylum', 'class',
                     'order', 'family', 'genus', 'species']:
            assert rank in result.columns

    def test_mixed_rows(self):
        df = pd.DataFrame({'lineage': [
            '-_cellular organisms;d_Bacteria;p_Proteobacteria;g_Escherichia',
            '-_cellular organisms;d_Archaea;p_Euryarchaeota;g_Methanococcus',
        ]})
        result = _parser().parse_taxonomy_lineage(df)
        assert list(result['domain']) == ['Bacteria', 'Archaea']
        assert list(result['genus']) == ['Escherichia', 'Methanococcus']
