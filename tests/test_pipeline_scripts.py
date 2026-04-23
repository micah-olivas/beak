"""Unit tests for remote/pipeline_scripts.py — pure bash-string builders."""

import pytest

from beak.remote.pipeline import PipelineStep
from beak.remote.pipeline_scripts import (
    generate_search_commands,
    generate_filter_commands,
    generate_taxonomy_commands,
    generate_align_commands,
    generate_tree_commands,
    generate_embeddings_commands,
)


DB_BASE = "/srv/dbs"
STEP_DIR = "/remote/job/01_search"
REMOTE_PATH = "/remote/job"
INPUT = "input.fasta"


def _step(step_type, params):
    return PipelineStep(
        step_type=step_type,
        step_name=f"{step_type}_0",
        params=dict(params),  # copy — PipelineStep mutates via pop
    )


class TestGenerateSearchCommands:
    def _run(self, params):
        return generate_search_commands(
            _step('search', params), STEP_DIR, INPUT, "00_search_output", REMOTE_PATH, DB_BASE
        )

    def test_includes_core_mmseqs_stages(self):
        cmds = "\n".join(self._run({'database': 'uniref90'}))
        assert 'mmseqs createdb' in cmds
        assert 'mmseqs search' in cmds
        assert 'mmseqs convertalis' in cmds
        assert 'mmseqs createseqfiledb' in cmds

    def test_expands_known_database_to_base_path(self):
        cmds = "\n".join(self._run({'database': 'uniref90'}))
        assert DB_BASE in cmds

    def test_absolute_database_path_passes_through(self):
        cmds = "\n".join(self._run({'database': '/custom/path/db'}))
        assert '/custom/path/db' in cmds
        assert f"{DB_BASE}//" not in cmds  # wasn't joined to base

    def test_preset_parameters_are_merged_in(self):
        # The 'fast' preset should contribute at least one --flag to the command
        cmds = "\n".join(self._run({'database': 'uniref90', 'preset': 'fast'}))
        assert '--' in cmds

    def test_output_name_used_for_hit_fasta(self):
        cmds = "\n".join(self._run({'database': 'uniref90'}))
        assert '00_search_output.fasta' in cmds

    def test_cleanup_at_end(self):
        cmds = self._run({'database': 'uniref90'})
        assert any('rm -rf' in line for line in cmds)


class TestGenerateFilterCommands:
    def _run(self, params):
        return generate_filter_commands(
            _step('filter', params), STEP_DIR, INPUT, REMOTE_PATH
        )

    def test_writes_python_script_via_heredoc(self):
        cmds = "\n".join(self._run({'size': [50, 500]}))
        assert 'cat >' in cmds and "<< 'EOF'" in cmds
        assert 'python3' in cmds

    def test_size_filter_emits_length_predicate(self):
        cmds = "\n".join(self._run({'size': [100, 400]}))
        assert '100 <= len(r.seq) <= 400' in cmds

    def test_motif_filter_uses_re_search(self):
        cmds = "\n".join(self._run({'motif': 'GXGXXG'}))
        assert "re.search(r'GXGXXG'" in cmds

    def test_deduplicate_emits_seen_set(self):
        cmds = "\n".join(self._run({'deduplicate': True}))
        assert 'seen = set()' in cmds

    def test_max_sequences_emits_sort_and_slice(self):
        cmds = "\n".join(self._run({'max_sequences': 1000}))
        assert 'sorted(' in cmds and '[:1000]' in cmds

    def test_writes_filtered_fasta_output(self):
        cmds = "\n".join(self._run({}))
        assert f"{STEP_DIR}/filtered.fasta" in cmds


class TestGenerateTaxonomyCommands:
    def _run(self, params):
        return generate_taxonomy_commands(
            _step('taxonomy', params), STEP_DIR, INPUT, REMOTE_PATH, DB_BASE
        )

    def test_core_mmseqs_taxonomy_stages(self):
        cmds = "\n".join(self._run({'database': 'uniprotkb'}))
        assert 'mmseqs createdb' in cmds
        assert 'mmseqs taxonomy' in cmds
        assert 'mmseqs createtsv' in cmds

    def test_tax_lineage_flag_emitted_by_default(self):
        cmds = "\n".join(self._run({'database': 'uniprotkb'}))
        assert '--tax-lineage 1' in cmds

    def test_tax_lineage_can_be_disabled(self):
        cmds = "\n".join(self._run({'database': 'uniprotkb', 'tax_lineage': False}))
        assert '--tax-lineage 1' not in cmds


class TestGenerateAlignCommands:
    def _run(self, params):
        return generate_align_commands(
            _step('align', params), STEP_DIR, INPUT, REMOTE_PATH
        )

    def test_default_algorithm_is_clustalo(self):
        cmds = "\n".join(self._run({}))
        assert 'clustalo' in cmds

    def test_mafft_dispatch(self):
        cmds = "\n".join(self._run({'algorithm': 'mafft'}))
        assert 'mafft' in cmds

    def test_muscle_dispatch(self):
        cmds = "\n".join(self._run({'algorithm': 'muscle'}))
        assert 'muscle' in cmds

    def test_seqkit_fallback_awk_present(self):
        # The pre-filter must work whether or not seqkit is installed
        cmds = "\n".join(self._run({}))
        assert 'command -v seqkit' in cmds
        assert 'awk' in cmds


class TestGenerateTreeCommands:
    def _run(self, params):
        return generate_tree_commands(
            _step('tree', params), STEP_DIR, INPUT, REMOTE_PATH
        )

    def test_iqtree_fallback_chain(self):
        cmds = "\n".join(self._run({}))
        assert 'iqtree2' in cmds
        assert 'iqtree' in cmds  # fallback branch

    def test_missing_iqtree_emits_placeholder(self):
        cmds = "\n".join(self._run({}))
        assert "'(iqtree not available)'" in cmds
        assert 'tree.nwk' in cmds


class TestGenerateEmbeddingsCommands:
    def _run(self, params):
        return generate_embeddings_commands(
            _step('embeddings', params), STEP_DIR, INPUT, REMOTE_PATH, '/remote'
        )

    def test_docker_compose_exec_used(self):
        cmds = "\n".join(self._run({}))
        assert 'docker compose' in cmds and 'exec' in cmds

    def test_project_name_is_consistent_across_users(self):
        # All beak users must land on the same docker compose project so
        # `exec` hits the shared service regardless of working directory.
        cmds = "\n".join(self._run({}))
        assert '--project-name beak' in cmds

    def test_custom_docker_dir_used_when_provided(self):
        from beak.remote.pipeline_scripts import generate_embeddings_commands
        cmds = "\n".join(generate_embeddings_commands(
            _step('embeddings', {}), STEP_DIR, INPUT, REMOTE_PATH,
            '/remote', docker_dir='/srv/beak_docker', project_name='beak',
        ))
        assert 'cd /srv/beak_docker' in cmds

    def test_default_model_is_esm2(self):
        cmds = "\n".join(self._run({}))
        assert 'esm2_t33_650M_UR50D' in cmds

    def test_custom_model_propagates(self):
        cmds = "\n".join(self._run({'model': 'esm2_t6_8M_UR50D'}))
        assert 'esm2_t6_8M_UR50D' in cmds

    def test_include_mean_is_default(self):
        cmds = "\n".join(self._run({}))
        assert '--include-mean' in cmds

    def test_include_per_tok_off_by_default(self):
        cmds = "\n".join(self._run({}))
        assert '--include-per-tok' not in cmds

    def test_include_per_tok_can_be_enabled(self):
        cmds = "\n".join(self._run({'include_per_tok': True}))
        assert '--include-per-tok' in cmds
