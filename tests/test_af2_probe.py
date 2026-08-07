"""Unit tests for the AF2/ColabFold health probe parser.

Pure parsing tests — no SSH, no network, matching the _parse_load_probe
tests in test_cli.py.
"""

import pytest

from beak.remote.af2 import (
    _build_discovery_script,
    _build_probe_script,
    _parse_af2_probe,
    plan_af2_setup,
    render_wrapper,
    CUDNN_PACKAGE,
    EXPECTED_CUDNN_LIBS,
    MIN_WEIGHT_FILES,
    WRAPPER_VERSION,
)


def probe_output(**fields) -> str:
    """Render a probe stdout blob from keyword fields."""
    return ''.join(f"{k}\t{v}\n" for k, v in fields.items())


HEALTHY = dict(
    binary='/home/u/bin/colabfold_batch',
    binary_source='wrapper',
    weights=20,
    numpy_default='1.26.4',
    numpy_isolated='1.26.4',
    stack_default='ok',
    stack_isolated='ok',
    binary_runs='ok',
    wrapper_preload='yes',
    wrapper_nousersite='yes',
    cudnn_libs=EXPECTED_CUDNN_LIBS,
    gpu_total=2,
    gpu_free=2,
    gpu_mem_mb=11264,
    backend_bare='gpu',
    backend_override='gpu',
)


def levels(report):
    return {i['level'] for i in report['issues']}


def messages(report):
    return ' | '.join(i['message'] for i in report['issues'])


class TestHealthyInstall:
    def test_clean_install_reports_ok_with_no_issues(self):
        report = _parse_af2_probe(probe_output(**HEALTHY))
        assert report['ok'] is True
        assert report['issues'] == []
        assert report['installed'] is True
        assert report['binary'] == '/home/u/bin/colabfold_batch'
        assert report['gpu_free'] == 2

    def test_a_wrapper_that_sets_the_env_is_not_flagged(self):
        """The bare interpreter being shadowed must not fail a wrapped install.

        This is the regression that matters: a correct wrapper repairs the
        environment at call time, so judging the raw interpreter alone would
        flag a working install as broken forever.
        """
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'stack_default': 'fail',
               'numpy_default': '2.2.6', 'binary_runs': 'ok'}))
        assert report['ok'] is True
        assert report['issues'] == []

    def test_shadowing_warns_when_the_entry_point_does_not_isolate(self):
        """Same state, but nothing guarantees it keeps working."""
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'stack_default': 'fail', 'numpy_default': '2.2.6',
               'wrapper_nousersite': 'no'}))
        assert report['ok'] is True
        assert levels(report) == {'warn'}
        assert 'does not do it itself' in messages(report)


class TestMissingInstall:
    def test_no_binary_is_an_error_and_short_circuits(self):
        report = _parse_af2_probe(probe_output(
            binary='-', binary_source='-', weights=0))
        assert report['installed'] is False
        assert report['ok'] is False
        assert len(report['issues']) == 1
        assert 'not found' in report['issues'][0]['message']

    def test_empty_output_degrades_to_not_installed(self):
        report = _parse_af2_probe('')
        assert report['installed'] is False
        assert report['ok'] is False

    def test_incomplete_weights_download_is_an_error(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'weights': MIN_WEIGHT_FILES - 1}))
        assert report['ok'] is False
        assert 'weight files' in messages(report)


class TestEnvironmentHealth:
    def test_user_site_shadowing_that_breaks_startup_is_an_error(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'binary_source': 'install', 'binary_runs': 'fail',
               'stack_default': 'fail', 'numpy_default': '2.2.6'}))
        assert report['ok'] is False
        assert 'fails to start' in messages(report)
        # The remedy must name the actual fix, not just the symptom.
        fixes = ' '.join(i['fix'] for i in report['issues'])
        assert 'PYTHONNOUSERSITE' in fixes

    def test_broken_env_outranks_shadowing_diagnosis(self):
        """If even the isolated import fails, the env itself is broken."""
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'stack_isolated': 'fail', 'stack_default': 'fail',
               'binary_runs': 'fail'}))
        assert report['ok'] is False
        assert 'reinstall' in ' '.join(i['fix'] for i in report['issues'])
        assert 'fails to start' not in messages(report)

    def test_bare_conda_entry_point_warns_but_stays_ok(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'binary_source': 'install',
               'wrapper_preload': 'no', 'wrapper_nousersite': 'no'}))
        assert report['ok'] is True
        assert levels(report) == {'warn'}
        assert 'bare conda entry point' in messages(report)


class TestGpuHealth:
    def test_gpu_unusable_even_with_override_is_an_error(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'backend_bare': 'cpu', 'backend_override': 'cpu'}))
        assert report['ok'] is False
        assert 'even with the cuDNN override' in messages(report)

    def test_gpu_only_via_override_warns_about_the_silent_fallback(self):
        """The exact silent-regression case: works, but only if configured."""
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'backend_bare': 'cpu', 'backend_override': 'gpu',
               'wrapper_preload': 'no'}))
        assert report['ok'] is True
        assert levels(report) == {'warn'}
        assert 'LD_PRELOAD' in messages(report) or 'override' in messages(report)

    def test_that_warning_clears_once_the_wrapper_exports_the_override(self):
        """A warning that can never be cleared is noise, not signal.

        With the entry point exporting LD_PRELOAD itself, GPU-only-via-override
        is a settled configuration rather than an outstanding problem.
        """
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'backend_bare': 'cpu', 'backend_override': 'gpu',
               'wrapper_preload': 'yes'}))
        assert report['ok'] is True
        assert report['issues'] == []

    def test_cpu_fallback_with_no_override_installed_is_an_error(self):
        # No override libraries at all, so the probe emits no
        # backend_override line to compare against.
        fields = {k: v for k, v in HEALTHY.items() if k != 'backend_override'}
        fields.update(backend_bare='cpu', cudnn_libs=0)
        report = _parse_af2_probe(probe_output(**fields))
        assert report['ok'] is False
        assert 'no cuDNN override' in messages(report)

    def test_partial_cudnn_override_warns(self):
        report = _parse_af2_probe(probe_output(**{**HEALTHY, 'cudnn_libs': 3}))
        assert report['ok'] is True
        assert f'3/{EXPECTED_CUDNN_LIBS}' in messages(report)

    def test_gpu_checks_are_skipped_on_a_cpu_only_host(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'gpu_total': 0, 'gpu_free': 0, 'cudnn_libs': 0,
               'backend_bare': 'cpu', 'backend_override': 'cpu'}))
        assert report['ok'] is True
        assert report['issues'] == []


class TestShrZionRegression:
    """The literal broken state observed on shr-zion, 2026-08-07.

    Both real defects must be reported, each with its remedy.
    """

    BROKEN = dict(
        binary='/home/mbolivas/localcolabfold/colabfold-conda/bin/colabfold_batch',
        binary_source='install',
        weights=20,
        numpy_default='2.2.6',
        numpy_isolated='1.26.4',
        stack_default='fail',
        stack_isolated='ok',
        binary_runs='fail',
        cudnn_libs=0,
        gpu_total=2,
        gpu_free=2,
        backend_bare='cpu',
    )

    def test_both_real_defects_are_reported(self):
        report = _parse_af2_probe(probe_output(**self.BROKEN))
        assert report['ok'] is False
        text = messages(report)
        assert 'fails to start' in text          # numpy shadowing
        assert 'no cuDNN override' in text       # silent CPU fallback
        assert '2.2.6' in text and '1.26.4' in text

    def test_the_repaired_state_is_completely_clean(self):
        """After the wrapper + cuDNN override, every issue must clear.

        The bare interpreter is still shadowed and bare JAX still lands on
        CPU — permanently. If those kept warning, the probe could never
        reach a clean state on this machine and would train the reader to
        ignore it.
        """
        repaired = {**self.BROKEN, 'binary_source': 'wrapper',
                    'binary_runs': 'ok', 'cudnn_libs': EXPECTED_CUDNN_LIBS,
                    'backend_bare': 'cpu', 'backend_override': 'gpu',
                    'wrapper_preload': 'yes', 'wrapper_nousersite': 'yes'}
        report = _parse_af2_probe(probe_output(**repaired))
        assert report['ok'] is True
        assert report['issues'] == []


class TestParsingRobustness:
    def test_malformed_lines_are_skipped_not_guessed(self):
        noisy = ("garbage without a tab\n"
                 + probe_output(**HEALTHY)
                 + "\n\ntrailing noise\n")
        report = _parse_af2_probe(noisy)
        assert report['ok'] is True
        assert report['binary_source'] == 'wrapper'

    def test_non_numeric_counts_fall_back_to_zero(self):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'weights': 'n/a', 'gpu_total': 'oops'}))
        assert report['weights'] == 0
        assert report['gpu_total'] == 0

    @pytest.mark.parametrize('placeholder', ['-', ''])
    def test_placeholder_values_read_as_absent(self, placeholder):
        report = _parse_af2_probe(probe_output(
            **{**HEALTHY, 'backend_override': placeholder}))
        assert report['backend_override'] is None


class TestProbeScript:
    def test_paths_are_substituted(self):
        script = _build_probe_script('/opt/cf', '/opt/cudnn', deep=False)
        assert 'AF2_HOME="/opt/cf"' in script
        assert 'CUDNN_DIR="/opt/cudnn"' in script
        assert '@AF2_HOME@' not in script

    def test_deep_flag_controls_the_expensive_backend_probe(self):
        cheap = _build_probe_script('/opt/cf', '/opt/cudnn', deep=False)
        deep = _build_probe_script('/opt/cf', '/opt/cudnn', deep=True)
        assert 'import jax' not in cheap
        assert 'import jax' in deep
        assert 'backend_bare' in deep and 'backend_override' in deep

    def test_health_check_never_preallocates_gpu_memory(self):
        """A probe must not reserve ~75% of a card on a shared box."""
        deep = _build_probe_script('/opt/cf', '/opt/cudnn', deep=True)
        assert deep.count('XLA_PYTHON_CLIENT_PREALLOCATE=false') == 2

    def test_discovery_requires_the_complete_cudnn_set(self):
        """A partial directory would mix 8.9 and the stale in-env 8.2.1."""
        script = _build_discovery_script(['/a'], ['/b'], '$HOME/bin')
        assert f'-eq {EXPECTED_CUDNN_LIBS}' in script


class TestWrapperRendering:
    def test_wrapper_exports_the_two_required_fixes(self):
        script = render_wrapper('/opt/cf', '/opt/cudnn', 'colabfold_batch')
        assert 'PYTHONNOUSERSITE=1' in script
        assert 'LD_PRELOAD' in script

    def test_wrapper_does_not_disable_preallocation(self):
        """ColabFold oversubscribes GPU memory on purpose; don't undercut it.

        batch.py sets TF_FORCE_UNIFIED_MEMORY=1 and MEM_FRACTION=4.0, asking
        JAX for ~4x the card and spilling into host RAM. That is what lets a
        long sequence run on a small GPU, so forcing PREALLOCATE=false in the
        wrapper would cap maximum sequence length. The health probe still
        sets it — a diagnostic must stay polite; a real prediction must not
        be crippled.
        """
        script = render_wrapper('/opt/cf', '/opt/cudnn', 'colabfold_batch')
        assert 'export XLA_PYTHON_CLIENT_PREALLOCATE' not in script

    def test_wrapper_uses_preload_not_library_path(self):
        """LD_LIBRARY_PATH cannot work here: RPATH is searched first."""
        script = render_wrapper('/opt/cf', '/opt/cudnn', 'colabfold_batch')
        assert 'export LD_LIBRARY_PATH' not in script

    def test_wrapper_execs_the_requested_entry_point(self):
        script = render_wrapper('/opt/cf', '/opt/cudnn', 'colabfold_search')
        assert 'exec "/opt/cf/colabfold-conda/bin/colabfold_search" "$@"' in script

    def test_wrapper_carries_a_detectable_version_marker(self):
        script = render_wrapper('/opt/cf', '/opt/cudnn', 'colabfold_batch')
        assert f'beak-af2-wrapper v{WRAPPER_VERSION}' in script

    def test_wrapper_tolerates_a_missing_cudnn_directory(self):
        """Wrapper must still run (on CPU) rather than break outright."""
        script = render_wrapper('/opt/cf', '/nonexistent', 'colabfold_batch')
        assert 'if [ -d "$L" ]' in script


class TestSetupPlanning:
    BASE = {'af2_home': '/opt/cf', 'cudnn_dir': '/srv/beak/cudnn',
            'wrapper_version': None, 'writable_home': True}

    def actions(self, plan):
        return [s['action'] for s in plan['steps']]

    def test_a_shared_cudnn_copy_is_reused_not_redownloaded(self):
        """The multi-user win: 700 MB once, not once per user."""
        plan = plan_af2_setup(self.BASE)
        assert 'reuse_cudnn' in self.actions(plan)
        assert 'install_cudnn' not in self.actions(plan)
        assert plan['cudnn_dir'] == '/srv/beak/cudnn'

    def test_missing_cudnn_falls_back_to_a_per_user_download(self):
        plan = plan_af2_setup({**self.BASE, 'cudnn_dir': None})
        assert 'install_cudnn' in self.actions(plan)
        assert plan['cudnn_dir'].endswith('/nvidia/cudnn/lib')

    def test_no_install_found_blocks_with_actionable_advice(self):
        plan = plan_af2_setup({**self.BASE, 'af2_home': None})
        assert plan['blocked']
        assert '--home' in plan['blocked']
        assert plan['steps'] == []

    def test_unwritable_home_blocks_rather_than_failing_midway(self):
        plan = plan_af2_setup({**self.BASE, 'cudnn_dir': None,
                               'writable_home': False})
        assert plan['blocked']
        assert plan['steps'] == []

    def test_current_wrapper_is_left_alone(self):
        """Idempotence: a second run must be a no-op."""
        plan = plan_af2_setup({**self.BASE,
                               'wrapper_version': WRAPPER_VERSION})
        assert 'write_wrapper' not in self.actions(plan)

    def test_outdated_wrapper_is_upgraded(self):
        plan = plan_af2_setup({**self.BASE, 'wrapper_version': 0})
        assert 'write_wrapper' in self.actions(plan)
        detail = ' '.join(s['detail'] for s in plan['steps'])
        assert f'v0 -> v{WRAPPER_VERSION}' in detail

    def test_the_pinned_cudnn_is_an_8x_build_for_cuda_11(self):
        """cuDNN 9 ships libcudnn.so.9; jaxlib 0.4.23 links .so.8."""
        assert 'cu11' in CUDNN_PACKAGE and '==8.' in CUDNN_PACKAGE
