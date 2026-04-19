"""Tests for the UniProt REST API client."""

import pytest
from beak.api.uniprot import _build_query_url, _parse_link_header


class TestBuildQueryUrl:
    def test_basic_url(self):
        url = _build_query_url('PF00069', ['accession', 'protein_name'], 500)
        assert 'pfam-PF00069' in url
        assert 'protein_name' in url
        assert 'size=500' in url
        assert 'format=json' in url

    def test_with_taxonomy_fields(self):
        fields = ['accession', 'protein_name', 'organism_name', 'organism_id']
        url = _build_query_url('PF00069', fields, 500)
        assert 'organism_name' in url
        assert 'organism_id' in url

    def test_with_lineage_fields(self):
        fields = ['accession', 'protein_name', 'organism_name',
                  'organism_id', 'lineage']
        url = _build_query_url('PF00069', fields, 500)
        assert 'lineage' in url

    def test_with_cursor(self):
        url = _build_query_url('PF00069', ['accession'], 500, cursor='abc123')
        assert 'cursor=abc123' in url

    def test_without_cursor(self):
        url = _build_query_url('PF00069', ['accession'], 500)
        assert 'cursor' not in url


class TestParseLinkHeader:
    def test_valid_next_link(self):
        header = '<https://rest.uniprot.org/uniprotkb/search?cursor=abc123&size=500>; rel="next"'
        result = _parse_link_header(header)
        assert result == 'https://rest.uniprot.org/uniprotkb/search?cursor=abc123&size=500'

    def test_empty_header(self):
        assert _parse_link_header('') is None

    def test_none_header(self):
        assert _parse_link_header(None) is None

    def test_no_next_rel(self):
        header = '<https://example.com>; rel="prev"'
        assert _parse_link_header(header) is None

    def test_multiple_links(self):
        header = (
            '<https://example.com/prev>; rel="prev", '
            '<https://example.com/next?cursor=xyz>; rel="next"'
        )
        result = _parse_link_header(header)
        assert result == 'https://example.com/next?cursor=xyz'
