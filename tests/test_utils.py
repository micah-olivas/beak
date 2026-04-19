"""Tests for beak.utils — pure functions, no I/O."""

import pandas as pd
import pytest
from pathlib import Path

from beak.utils import (
    parse_uniprot_header, parse_fasta_headers, add_sequence_properties,
    fetch_uniprot, generate_readable_name,
)


class TestParseUniprotHeader:
    def test_standard_swissprot(self):
        result = parse_uniprot_header('sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein')
        assert result['db'] == 'sp'
        assert result['accession'] == 'P0DTC2'
        assert result['name'] == 'SPIKE_SARS2'
        assert result['description'] == 'Spike glycoprotein'

    def test_trembl(self):
        result = parse_uniprot_header('tr|A0A0A0|NAME_HUMAN Some protein')
        assert result['db'] == 'tr'
        assert result['accession'] == 'A0A0A0'
        assert result['name'] == 'NAME_HUMAN'

    def test_no_description(self):
        result = parse_uniprot_header('sp|P12345|MY_PROT')
        assert result['accession'] == 'P12345'
        assert result['description'] == ''

    def test_non_uniprot(self):
        result = parse_uniprot_header('some_random_header')
        assert result['db'] is None
        assert result['accession'] is None
        assert result['name'] == 'some_random_header'


class TestParseFastaHeaders:
    def test_uniprot_parsing(self):
        df = pd.DataFrame({'target': ['sp|P12345|MY_PROT Some protein']})
        result = parse_fasta_headers(df, header_col='target', parse_uniprot=True)
        assert 'accession' in result.columns
        assert result.iloc[0]['accession'] == 'P12345'

    def test_simple_parsing(self):
        df = pd.DataFrame({'target': ['seq1 A protein']})
        result = parse_fasta_headers(df, header_col='target', parse_uniprot=False)
        assert result.iloc[0]['header_id'] == 'seq1'
        assert result.iloc[0]['header_description'] == 'A protein'


class TestFetchUniprot:
    def test_fetch_valid_accession(self, tmp_path):
        path = fetch_uniprot('P0DTC2', output_dir=str(tmp_path))
        assert Path(path).exists()
        content = Path(path).read_text()
        assert content.startswith('>')
        assert 'SPIKE_SARS2' in content or 'P0DTC2' in content

    def test_fetch_invalid_accession(self):
        with pytest.raises(ValueError, match="not found"):
            fetch_uniprot('XXXXXXXXXX_INVALID')


class TestAddSequenceProperties:
    def test_basic_properties(self):
        df = pd.DataFrame({'sequence': ['MKTAYIAK']})
        result = add_sequence_properties(df, sequence_col='sequence')
        assert 'length' in result.columns
        assert result.iloc[0]['length'] == 8
        assert result.iloc[0]['molecular_weight'] > 0
        assert result.iloc[0]['isoelectric_point'] > 0

    def test_empty_sequence(self):
        df = pd.DataFrame({'sequence': ['']})
        result = add_sequence_properties(df, sequence_col='sequence')
        assert result.iloc[0]['length'] == 0

    def test_nan_sequence(self):
        df = pd.DataFrame({'sequence': [None]})
        result = add_sequence_properties(df, sequence_col='sequence')
        assert result.iloc[0]['length'] == 0


class TestGenerateReadableName:
    def test_format(self):
        name = generate_readable_name()
        parts = name.split('-')
        assert len(parts) == 3, f"Expected adjective-verb-noun, got: {name}"

    def test_uniqueness(self):
        names = {generate_readable_name() for _ in range(100)}
        # With ~216k combinations, 100 samples should all be unique
        assert len(names) == 100
