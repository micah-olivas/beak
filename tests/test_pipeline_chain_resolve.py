"""Tests for the pipeline-step chaining resolver used by `beak embeddings --from-job`."""

import pytest
import click

from beak.cli.submit import _resolve_pipeline_fasta


RP = "/remote/jobs/ABCD1234"


def _job(*step_types, params_by_type=None):
    """Build a minimal pipeline job_info dict for testing.

    params_by_type: optional dict of {step_type: {param_key: value}}
    """
    params_by_type = params_by_type or {}
    return {
        'remote_path': RP,
        'steps': [
            {'type': t, 'params': params_by_type.get(t, {})}
            for t in step_types
        ],
    }


class TestResolvePipelineFasta:
    def test_align_last_picks_alignment_fasta(self):
        path = _resolve_pipeline_fasta(_job('search', 'align'))
        assert path == f"{RP}/02_align/alignment.fasta"

    def test_align_respects_output_format(self):
        path = _resolve_pipeline_fasta(_job(
            'search', 'align',
            params_by_type={'align': {'output_format': 'phylip'}},
        ))
        assert path == f"{RP}/02_align/alignment.phylip"

    def test_filter_last_picks_filtered_fasta(self):
        path = _resolve_pipeline_fasta(_job('search', 'filter'))
        assert path == f"{RP}/02_filter/filtered.fasta"

    def test_search_only_picks_hit_output(self):
        path = _resolve_pipeline_fasta(_job('search'))
        assert path == f"{RP}/01_search/01_search_output.fasta"

    def test_walks_backwards_past_non_fasta_steps(self):
        # tree/taxonomy/embeddings at the end aren't FASTA-producing; should
        # fall back to the last step that IS one.
        path = _resolve_pipeline_fasta(_job('search', 'align', 'tree'))
        assert path == f"{RP}/02_align/alignment.fasta"

        path = _resolve_pipeline_fasta(_job('search', 'align', 'taxonomy'))
        assert path == f"{RP}/02_align/alignment.fasta"

        path = _resolve_pipeline_fasta(_job('search', 'filter', 'taxonomy', 'tree'))
        assert path == f"{RP}/02_filter/filtered.fasta"

    def test_correct_step_index_for_deeper_pipeline(self):
        # Search is step 3; output path must reflect NN=03 in both path
        # segments.
        path = _resolve_pipeline_fasta(_job('taxonomy', 'taxonomy', 'search'))
        assert path == f"{RP}/03_search/03_search_output.fasta"

    def test_no_fasta_producing_step_raises(self):
        with pytest.raises(click.BadParameter, match="no FASTA output"):
            _resolve_pipeline_fasta(_job('taxonomy', 'tree'))

    def test_missing_steps_raises(self):
        with pytest.raises(click.BadParameter, match="no steps"):
            _resolve_pipeline_fasta({'remote_path': RP, 'steps': []})

    def test_missing_remote_path_raises(self):
        with pytest.raises(click.BadParameter, match="no steps or remote_path"):
            _resolve_pipeline_fasta({'steps': [{'type': 'search', 'params': {}}]})
