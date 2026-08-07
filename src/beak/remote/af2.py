"""Health probe for a remote AlphaFold2 / localcolabfold install.

AF2 needs a bespoke probe rather than an entry in ``VERSION_PROBES`` for two
reasons, both learned the hard way on a real install:

1. **It is usually not on PATH.** localcolabfold installs into its own conda
   prefix and does not add itself to the shell profile, so the
   non-interactive PATH that Fabric's ``conn.run`` inherits does not contain
   ``colabfold_batch``. A plain ``command -v`` probe reports "missing" on a
   perfectly good install.

2. **Its worst failure mode is silent.** When JAX cannot load a new enough
   cuDNN it prints one warning and falls back to CPU. Nothing fails, nothing
   exits non-zero — predictions just get one to two orders of magnitude
   slower. A liveness check that only asks "does the binary exist?" would
   have happily reported OK while the GPU sat idle. So this probe asks what
   backend JAX actually resolves to.

The environment fixes this probe checks for are unusual enough to be worth
recording. ``PYTHONNOUSERSITE=1`` is needed because a conda prefix does not
isolate ``~/.local`` user site-packages, so a stray user-level numpy silently
overrides the env's pinned one and breaks the ABI that ``ml_dtypes`` compiled
against. And the cuDNN override has to be applied with ``LD_PRELOAD``, not
``LD_LIBRARY_PATH``: both the conda ``python`` and jaxlib's ``xla_extension.so``
carry a ``DT_RPATH``, which glibc searches *before* ``LD_LIBRARY_PATH``, so the
stale in-env cuDNN wins no matter what ``LD_LIBRARY_PATH`` says.

Parsing is split into a pure ``_parse_af2_probe`` so it is unit-testable
without SSH, matching ``_parse_load_probe`` in ``base.py``.
"""

from typing import Dict, Optional

from fabric import Connection


# Defaults match a stock localcolabfold layout. Both are overridable via an
# [af2] section in ~/.beak/config.toml so this is not welded to one machine.
DEFAULT_AF2_HOME = "$HOME/localcolabfold"
DEFAULT_CUDNN_DIR = "$HOME/cudnn89/nvidia/cudnn/lib"

# cuDNN 8.x ships the main library plus six companions
# (ops/cnn/adv x infer/train). The main library dlopens its siblings by
# soname at runtime, so a partial override still resolves the missing ones
# through RPATH to the stale in-env copy. All seven or it does not count.
EXPECTED_CUDNN_LIBS = 7

# AF2 ships 5 base models; a complete localcolabfold download (monomer +
# ptm + multimer v2/v3) is far more. Fewer than 5 means an interrupted
# download, not a variant layout.
MIN_WEIGHT_FILES = 5

_PROBE = r'''
AF2_HOME="@AF2_HOME@"
CUDNN_DIR="@CUDNN_DIR@"
PY="$AF2_HOME/colabfold-conda/bin/python3.10"

# Prefer a wrapper: it can carry the env fixes, whereas the bare conda entry
# point relies on the caller getting them right every time.
BIN=""; SRC="-"
if command -v colabfold_batch >/dev/null 2>&1; then
  BIN=$(command -v colabfold_batch); SRC="path"
elif [ -x "$HOME/bin/colabfold_batch" ]; then
  BIN="$HOME/bin/colabfold_batch"; SRC="wrapper"
elif [ -x "$AF2_HOME/colabfold-conda/bin/colabfold_batch" ]; then
  BIN="$AF2_HOME/colabfold-conda/bin/colabfold_batch"; SRC="install"
fi
printf 'binary\t%s\n' "${BIN:--}"
printf 'binary_source\t%s\n' "$SRC"
printf 'weights\t%s\n' "$(ls "$AF2_HOME"/colabfold/params/params_model_*.npz 2>/dev/null | wc -l | tr -d ' ')"
printf 'cudnn_libs\t%s\n' "$(ls "$CUDNN_DIR"/libcudnn*.so.8 2>/dev/null | wc -l | tr -d ' ')"

# Import ml_dtypes, not just numpy: numpy itself imports fine under a
# shadowing user-site install, and it is ml_dtypes (compiled against the
# env's numpy ABI) that actually explodes. This is the precise failure.
if [ -x "$PY" ]; then
  printf 'numpy_default\t%s\n' "$("$PY" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo -)"
  printf 'numpy_isolated\t%s\n' "$(PYTHONNOUSERSITE=1 "$PY" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo -)"
  printf 'stack_default\t%s\n' "$("$PY" -c 'import ml_dtypes' >/dev/null 2>&1 && echo ok || echo fail)"
  printf 'stack_isolated\t%s\n' "$(PYTHONNOUSERSITE=1 "$PY" -c 'import ml_dtypes' >/dev/null 2>&1 && echo ok || echo fail)"
else
  printf 'numpy_default\t-\nnumpy_isolated\t-\nstack_default\t-\nstack_isolated\t-\n'
fi

# The decisive environment check: does the binary work *as invoked*? A
# wrapper repairs the environment at call time, so probing the bare
# interpreter alone would flag a correctly-wrapped install as broken
# forever. --help exercises the full import chain.
if [ -n "$BIN" ]; then
  printf 'binary_runs\t%s\n' "$("$BIN" --help >/dev/null 2>&1 && echo ok || echo fail)"
else
  printf 'binary_runs\t-\n'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'gpu_total\t%s\n' "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  printf 'gpu_free\t%s\n' "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 500' | wc -l | tr -d ' ')"
else
  printf 'gpu_total\t0\ngpu_free\t0\n'
fi
'''

# Deep check: resolve JAX's actual backend, twice.
#
# Probing only *with* the cuDNN override would answer "could the GPU work?"
# while the silent failure we care about is "does it work as things are
# actually configured?" — an install whose wrapper forgets LD_PRELOAD looks
# healthy under an override-only probe and quietly runs every prediction on
# CPU. So: `bare` is what an unconfigured caller gets, `override` is the
# ceiling. Comparing them separates "GPU is broken" from "GPU works but the
# environment is not wired up".
#
# Split from the cheap probe because each jax import plus CUDA init costs
# seconds. PREALLOCATE=false so a health check never reserves ~75% of a
# shared card.
_PROBE_DEEP = r'''
if [ -x "$PY" ]; then
  BACKEND_CMD='import jax;print(jax.default_backend())'
  printf 'backend_bare\t%s\n' "$(PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$PY" -c "$BACKEND_CMD" 2>/dev/null | tail -1 || echo -)"
  PRE=$(ls "$CUDNN_DIR"/libcudnn*.so.8 2>/dev/null | tr '\n' ':' | sed 's/:$//')
  if [ -n "$PRE" ]; then
    printf 'backend_override\t%s\n' "$(LD_PRELOAD="$PRE" PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" -c "$BACKEND_CMD" 2>/dev/null | tail -1 || echo -)"
  fi
fi
'''


def _build_probe_script(af2_home: str, cudnn_dir: str, deep: bool) -> str:
    """Render the remote probe script. One SSH round trip for everything."""
    script = _PROBE
    if deep:
        script = script + _PROBE_DEEP
    return script.replace('@AF2_HOME@', af2_home).replace('@CUDNN_DIR@', cudnn_dir)


def _parse_af2_probe(stdout: str) -> Dict:
    """Turn the tab-separated probe output into a structured health report.

    Pure (no I/O) so it is unit-testable. Unknown or unparseable fields are
    omitted rather than guessed at; a field that never arrived reads as
    "not probed", which downstream renders as unknown rather than healthy.

    Returns a dict with the raw observations plus an ``issues`` list of
    ``{level, message, fix}`` and an ``ok`` flag (True when no issue is at
    error level).
    """
    raw = {}
    for line in (stdout or '').splitlines():
        if '\t' not in line:
            continue
        key, _, value = line.partition('\t')
        key, value = key.strip(), value.strip()
        if key:
            raw[key] = value

    def _int(key: str) -> int:
        v = raw.get(key, '')
        return int(v) if v.isdigit() else 0

    def _str(key: str) -> Optional[str]:
        v = raw.get(key)
        return None if v in (None, '', '-') else v

    report = {
        'installed': _str('binary') is not None,
        'binary': _str('binary'),
        'binary_source': _str('binary_source'),
        'weights': _int('weights'),
        'numpy_default': _str('numpy_default'),
        'numpy_isolated': _str('numpy_isolated'),
        'stack_default': _str('stack_default'),
        'stack_isolated': _str('stack_isolated'),
        'binary_runs': _str('binary_runs'),
        'cudnn_libs': _int('cudnn_libs'),
        'gpu_total': _int('gpu_total'),
        'gpu_free': _int('gpu_free'),
        'backend_bare': _str('backend_bare'),
        'backend_override': _str('backend_override'),
        'issues': [],
    }

    def issue(level: str, message: str, fix: str):
        report['issues'].append({'level': level, 'message': message, 'fix': fix})

    if not report['installed']:
        # Everything downstream is meaningless without a binary, so report
        # the one actionable thing and stop.
        issue('error', 'colabfold_batch not found',
              'Install localcolabfold, or set [af2] home in ~/.beak/config.toml')
        report['ok'] = False
        return report

    if report['weights'] < MIN_WEIGHT_FILES:
        issue('error',
              f"only {report['weights']} AF2 weight files "
              f"(expected >= {MIN_WEIGHT_FILES})",
              'Re-run the localcolabfold weight download')

    # Environment health.
    #
    # binary_runs is the verdict, because it tests the binary the way beak
    # would actually call it: a wrapper that exports PYTHONNOUSERSITE makes a
    # shadowed bare interpreter irrelevant. stack_* explains *why*, and only
    # ever downgrades to a warning once the binary itself is known to work.
    if report['stack_isolated'] == 'fail':
        issue('error',
              "the env's numeric stack (ml_dtypes) cannot import even with "
              "user-site isolated",
              'The conda env itself is broken — reinstall localcolabfold')
    elif report['binary_runs'] == 'fail':
        detail = ''
        if report['numpy_default'] and report['numpy_isolated']:
            detail = (f" (user-site numpy {report['numpy_default']} shadows "
                      f"env numpy {report['numpy_isolated']})")
        issue('error',
              f'colabfold_batch fails to start{detail}',
              'Set PYTHONNOUSERSITE=1 — a ~/bin/colabfold_batch wrapper is '
              'the durable fix')
    elif report['stack_default'] == 'fail':
        # Runs, but only because the caller repairs the environment. Worth
        # surfacing: anything invoking the bare interpreter still breaks.
        issue('warn',
              'runs only because the caller isolates user site-packages; the '
              'bare interpreter is still shadowed',
              'Keep invoking through the wrapper, or clear the offending '
              '~/.local packages')

    # GPU health, only meaningful where a GPU actually exists and only
    # assessable when the deep probe ran.
    if report['gpu_total']:
        if report['cudnn_libs'] != EXPECTED_CUDNN_LIBS:
            issue('warn',
                  f"cuDNN override has {report['cudnn_libs']}/"
                  f"{EXPECTED_CUDNN_LIBS} libraries",
                  'pip install --target ~/cudnn89 nvidia-cudnn-cu11==8.9.6.50')

        bare, override = report['backend_bare'], report['backend_override']
        if override and override != 'gpu':
            issue('error',
                  f"{report['gpu_total']} GPU present but JAX resolves to "
                  f"{override} even with the cuDNN override — predictions "
                  "will be far slower",
                  'Check that the override libraries match what jaxlib was '
                  'built against')
        elif override == 'gpu' and bare and bare != 'gpu':
            # The silent-regression case: usable, but only if every caller
            # remembers the override.
            issue('warn',
                  'GPU works only with the cuDNN LD_PRELOAD override; without '
                  f'it JAX falls back to {bare}',
                  'Ensure the wrapper exports LD_PRELOAD with all 7 '
                  'libcudnn*.so.8 (LD_LIBRARY_PATH cannot work: RPATH is '
                  'searched first)')
        elif bare and bare != 'gpu' and not override:
            issue('error',
                  f"{report['gpu_total']} GPU present but JAX resolves to "
                  f"{bare}, and no cuDNN override is installed",
                  'pip install --target ~/cudnn89 nvidia-cudnn-cu11==8.9.6.50, '
                  'then LD_PRELOAD all 7 libcudnn*.so.8')

    if report['binary_source'] == 'install':
        issue('warn',
              'calling the bare conda entry point; env fixes are not applied '
              'automatically',
              'Create a ~/bin/colabfold_batch wrapper that exports '
              'PYTHONNOUSERSITE, LD_PRELOAD and XLA_PYTHON_CLIENT_PREALLOCATE')

    report['ok'] = not any(i['level'] == 'error' for i in report['issues'])
    return report


def probe_af2(conn: Connection,
              af2_home: Optional[str] = None,
              cudnn_dir: Optional[str] = None,
              deep: bool = False) -> Dict:
    """Probe the remote AF2/ColabFold install and report its health.

    Args:
        conn: Fabric SSH connection
        af2_home: localcolabfold prefix; defaults to the [af2] config
            section, then DEFAULT_AF2_HOME
        cudnn_dir: directory holding the cuDNN 8.9 override libraries
        deep: also resolve JAX's actual backend. Costs seconds (imports jax
            and initializes CUDA), so callers opt in.

    Returns:
        The dict described by _parse_af2_probe. A probe that cannot run at
        all yields a report with installed=False rather than raising, so
        `doctor` degrades to "not found" instead of blowing up.
    """
    if af2_home is None or cudnn_dir is None:
        try:
            from ..config import load_config
            section = load_config().get('af2', {})
        except Exception:
            section = {}
        af2_home = af2_home or section.get('home', DEFAULT_AF2_HOME)
        cudnn_dir = cudnn_dir or section.get('cudnn_dir', DEFAULT_CUDNN_DIR)

    script = _build_probe_script(af2_home, cudnn_dir, deep)
    try:
        # Generous timeout on deep: a cold CUDA init can take tens of seconds.
        result = conn.run(script, hide=True, warn=True,
                          timeout=180 if deep else 30)
        stdout = result.stdout or ''
    except Exception:
        stdout = ''
    return _parse_af2_probe(stdout)
