"""Unit tests for remote/align.py command builders — pure string builders."""

import pytest

from beak.remote.align import (
    ALGORITHMS,
    _CMD_BUILDERS,
    _build_clustalo_cmd,
    _build_mafft_cmd,
    _build_muscle_cmd,
)


IN = '/in/seqs.fasta'
OUT = '/out/aln.fasta'
LOG = '/out/tool.log'


class TestAlgorithmsTable:
    def test_registered_algorithms(self):
        assert set(ALGORITHMS) == {'clustalo', 'mafft', 'muscle'}

    def test_each_has_log_file_and_default_format(self):
        for name, info in ALGORITHMS.items():
            assert 'log_file' in info
            assert info['default_format'] in info['output_formats']

    def test_builders_cover_all_algorithms(self):
        assert set(_CMD_BUILDERS) == set(ALGORITHMS)


class TestClustaloCmd:
    def test_basic_invocation(self):
        cmd = _build_clustalo_cmd(IN, OUT, 'fasta', LOG, {})
        assert 'clustalo' in cmd
        assert f'-i {IN}' in cmd
        assert f'-o {OUT}' in cmd
        assert LOG in cmd

    def test_default_fasta_suppresses_outfmt_flag(self):
        cmd = _build_clustalo_cmd(IN, OUT, 'fasta', LOG, {})
        assert '--outfmt' not in cmd

    def test_non_fasta_format_adds_outfmt_flag(self):
        cmd = _build_clustalo_cmd(IN, OUT, 'phylip', LOG, {})
        assert '--outfmt=phylip' in cmd

    def test_params_get_long_flags_with_dashes(self):
        cmd = _build_clustalo_cmd(IN, OUT, 'fasta', LOG, {'max_iterations': 3})
        assert '--max-iterations=3' in cmd


class TestMafftCmd:
    def test_basic_invocation_writes_stdout_to_output(self):
        cmd = _build_mafft_cmd(IN, OUT, 'fasta', LOG, {})
        assert cmd.startswith('mafft')
        assert f'> {OUT}' in cmd
        assert f'2> {LOG}' in cmd
        assert IN in cmd

    def test_boolean_true_becomes_bare_flag(self):
        cmd = _build_mafft_cmd(IN, OUT, 'fasta', LOG, {'auto': True})
        assert '--auto' in cmd
        assert '--auto True' not in cmd  # boolean must not get a value

    def test_valued_param_gets_space_separated_value(self):
        cmd = _build_mafft_cmd(IN, OUT, 'fasta', LOG, {'thread': 8})
        assert '--thread 8' in cmd

    def test_underscore_to_dash_conversion(self):
        cmd = _build_mafft_cmd(IN, OUT, 'fasta', LOG, {'op_penalty': 1.5})
        assert '--op-penalty 1.5' in cmd


class TestMuscleCmd:
    def test_basic_invocation(self):
        cmd = _build_muscle_cmd(IN, OUT, 'fasta', LOG, {})
        assert 'muscle' in cmd
        assert f'-align {IN}' in cmd
        assert f'-output {OUT}' in cmd
        assert LOG in cmd

    def test_param_becomes_single_dash_flag_with_value(self):
        cmd = _build_muscle_cmd(IN, OUT, 'fasta', LOG, {'super5': True})
        assert '-super5 True' in cmd
