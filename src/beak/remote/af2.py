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

import base64
from typing import Callable, Dict, List, Optional

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

# Does the resolved entry point carry the environment fixes itself? A
# wrapper that exports these makes the bare interpreter's state irrelevant,
# which is what lets the corresponding warnings actually clear instead of
# nagging forever on a correctly configured install.
if [ -n "$BIN" ] && [ -f "$BIN" ]; then
  HEAD=$(head -c 65536 "$BIN" 2>/dev/null)
  printf 'wrapper_preload\t%s\n' "$(printf '%s' "$HEAD" | grep -q 'LD_PRELOAD' && echo yes || echo no)"
  printf 'wrapper_nousersite\t%s\n' "$(printf '%s' "$HEAD" | grep -q 'PYTHONNOUSERSITE' && echo yes || echo no)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'gpu_total\t%s\n' "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  printf 'gpu_free\t%s\n' "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 500' | wc -l | tr -d ' ')"
  # Smallest card, not the largest: VRAM is the binding constraint on what
  # will fit, and a job can land on any of them.
  printf 'gpu_mem_mb\t%s\n' "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 | tr -d ' ')"
else
  printf 'gpu_total\t0\ngpu_free\t0\ngpu_mem_mb\t0\n'
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


def _parse_kv(stdout: str) -> Dict[str, str]:
    """Collect ``key<TAB>value`` lines. Malformed lines are skipped."""
    out = {}
    for line in (stdout or '').splitlines():
        if '\t' not in line:
            continue
        key, _, value = line.partition('\t')
        key, value = key.strip(), value.strip()
        if key:
            out[key] = value
    return out


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
        'wrapper_preload': _str('wrapper_preload'),
        'wrapper_nousersite': _str('wrapper_nousersite'),
        'cudnn_libs': _int('cudnn_libs'),
        'gpu_total': _int('gpu_total'),
        'gpu_free': _int('gpu_free'),
        'gpu_mem_mb': _int('gpu_mem_mb'),
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
    elif (report['stack_default'] == 'fail'
            and report['wrapper_nousersite'] != 'yes'):
        # Runs today, but nothing in the entry point guarantees it keeps
        # working — the caller happened to isolate user site-packages. Once
        # the wrapper exports PYTHONNOUSERSITE itself this is settled, so the
        # warning clears rather than nagging forever. The shadowed numpy
        # versions stay visible as table detail either way.
        issue('warn',
              'runs only because the caller isolates user site-packages; the '
              'entry point does not do it itself',
              'Export PYTHONNOUSERSITE=1 from a ~/bin/colabfold_batch wrapper')

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
        elif (override == 'gpu' and bare and bare != 'gpu'
                and report['wrapper_preload'] != 'yes'):
            # The silent-regression case: usable, but only if every caller
            # remembers the override. An entry point that exports LD_PRELOAD
            # itself has settled this, so the warning clears.
            issue('warn',
                  'GPU works only with the cuDNN LD_PRELOAD override, and the '
                  f'entry point does not set it; bare JAX falls back to {bare}',
                  'Export LD_PRELOAD with all 7 libcudnn*.so.8 from a '
                  '~/bin/colabfold_batch wrapper (LD_LIBRARY_PATH cannot '
                  'work: RPATH is searched first)')
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


# ── Standardized setup ────────────────────────────────────────────
#
# The point of `beak setup af2` is that a second user on the same box does
# not repeat the archaeology the first one did. Two things make that work:
# discovery (find what already exists rather than assuming one layout) and
# reuse (a 700 MB cuDNN override copied per-user is pure waste on a shared
# filesystem — ten users is 7 GB of identical bytes).

# Searched in order; first hit wins. A site can prepend its own via the
# [af2] section of ~/.beak/config.toml.
AF2_HOME_CANDIDATES = [
    "$HOME/localcolabfold",
    "/srv/localcolabfold",
    "/opt/localcolabfold",
]
CUDNN_DIR_CANDIDATES = [
    "$HOME/cudnn89/nvidia/cudnn/lib",
    "/srv/beak/cudnn89/nvidia/cudnn/lib",
    "/opt/beak/cudnn89/nvidia/cudnn/lib",
]

# Pinned: jaxlib 0.4.23 is built against cuDNN 8.6, and cuDNN 9 ships
# libcudnn.so.9 while jaxlib links libcudnn.so.8 — so "latest" is actively
# wrong here. 8.9.6.50 is the newest 8.x for CUDA 11.
CUDNN_PACKAGE = "nvidia-cudnn-cu11==8.9.6.50"

# Bumped when the wrapper's content changes, so setup can tell an outdated
# wrapper from a current one instead of silently leaving a stale file.
# v2: stopped forcing XLA_PYTHON_CLIENT_PREALLOCATE=false, which fought
#     ColabFold's unified-memory oversubscription (see the template).
WRAPPER_VERSION = 2
WRAPPER_ENTRY_POINTS = ["colabfold_batch", "colabfold_search"]

_WRAPPER_TEMPLATE = r'''#!/usr/bin/env bash
# beak-af2-wrapper v@VERSION@ — managed by `beak setup af2`, do not hand-edit.
#
# LD_PRELOAD rather than LD_LIBRARY_PATH: the conda python and jaxlib's
# xla_extension.so both carry a DT_RPATH, which glibc searches *before*
# LD_LIBRARY_PATH, so the stale in-env cuDNN would win otherwise. All seven
# libraries are preloaded because the main one dlopens its siblings by
# soname at runtime.
L="@CUDNN_DIR@"
if [ -d "$L" ]; then
  export LD_PRELOAD="$(ls "$L"/libcudnn*.so.8 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
fi
# A conda prefix does not isolate ~/.local, so a stray user-level numpy
# silently overrides the env's pinned one and breaks the ABI ml_dtypes was
# compiled against.
export PYTHONNOUSERSITE=1
# Deliberately NOT setting XLA_PYTHON_CLIENT_PREALLOCATE here. ColabFold
# sets TF_FORCE_UNIFIED_MEMORY=1 and XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
# itself, asking JAX for ~4x the card's memory and spilling the excess into
# host RAM. That oversubscription is what lets a long sequence run at all on
# a small GPU, and disabling preallocation undercuts it. Being a polite
# neighbour is the health probe's job, not a real prediction's.
# Pass --disable-unified-memory to colabfold_batch if JAX errors instead.
exec "@AF2_HOME@/colabfold-conda/bin/@ENTRY@" "$@"
'''


def render_wrapper(af2_home: str, cudnn_dir: str, entry: str) -> str:
    """Render the wrapper script for one ColabFold entry point.

    Pure, so the generated script is unit-testable without SSH.
    """
    return (_WRAPPER_TEMPLATE
            .replace('@VERSION@', str(WRAPPER_VERSION))
            .replace('@CUDNN_DIR@', cudnn_dir)
            .replace('@AF2_HOME@', af2_home)
            .replace('@ENTRY@', entry))


def _build_discovery_script(af2_candidates, cudnn_candidates,
                            wrapper_dir: str) -> str:
    """Render the discovery probe: what already exists on this machine?"""
    af2_list = ' '.join(f'"{p}"' for p in af2_candidates)
    cudnn_list = ' '.join(f'"{p}"' for p in cudnn_candidates)
    return f'''
for p in {af2_list}; do
  if [ -x "$p/colabfold-conda/bin/colabfold_batch" ]; then
    printf 'af2_home\\t%s\\n' "$p"; break
  fi
done
# A candidate only counts with the complete set; a partial directory would
# resolve some sonames through RPATH to the stale copy and mix versions.
for p in {cudnn_list}; do
  n=$(ls "$p"/libcudnn*.so.8 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -eq {EXPECTED_CUDNN_LIBS} ]; then
    printf 'cudnn_dir\\t%s\\n' "$p"; break
  fi
done
printf 'wrapper_version\\t%s\\n' "$(grep -m1 -o 'beak-af2-wrapper v[0-9]*' \
    "{wrapper_dir}/colabfold_batch" 2>/dev/null | grep -o '[0-9]*$' || echo -)"
printf 'writable_home\\t%s\\n' "$([ -w "$HOME" ] && echo yes || echo no)"
'''


def discover_af2(conn: Connection,
                 af2_home: Optional[str] = None,
                 cudnn_dir: Optional[str] = None,
                 wrapper_dir: str = "$HOME/bin") -> Dict:
    """Find an existing AF2 install and cuDNN override on the remote.

    Configured paths are tried first, then the shared candidates. Returns
    ``{af2_home, cudnn_dir, wrapper_version, writable_home}`` with None for
    anything not found.
    """
    af2_candidates = ([af2_home] if af2_home else []) + AF2_HOME_CANDIDATES
    cudnn_candidates = ([cudnn_dir] if cudnn_dir else []) + CUDNN_DIR_CANDIDATES
    script = _build_discovery_script(af2_candidates, cudnn_candidates,
                                     wrapper_dir)
    try:
        result = conn.run(script, hide=True, warn=True, timeout=30)
        raw = _parse_kv(result.stdout or '')
    except Exception:
        raw = {}

    def _val(key):
        v = raw.get(key)
        return None if v in (None, '', '-') else v

    version = _val('wrapper_version')
    return {
        'af2_home': _val('af2_home'),
        'cudnn_dir': _val('cudnn_dir'),
        'wrapper_version': int(version) if (version or '').isdigit() else None,
        'writable_home': raw.get('writable_home') == 'yes',
    }


def plan_af2_setup(discovery: Dict, wrapper_dir: str = "$HOME/bin",
                   cudnn_target: str = "$HOME/cudnn89") -> Dict:
    """Decide what `beak setup af2` needs to do. Pure, so it is testable.

    Returns ``{steps, blocked, af2_home, cudnn_dir}``. Each step is
    ``{action, detail}``; ``blocked`` carries a reason when setup cannot
    proceed at all.
    """
    steps = []
    af2_home = discovery.get('af2_home')
    cudnn_dir = discovery.get('cudnn_dir')

    if not af2_home:
        return {
            'steps': [],
            'blocked': ('No localcolabfold install found. Install one, or '
                        'point beak at an existing one with '
                        '`beak setup af2 --home <path>`.'),
            'af2_home': None,
            'cudnn_dir': cudnn_dir,
        }

    if cudnn_dir:
        # The whole point of discovery: a shared copy means this user does
        # not download 700 MB of bytes that already exist on the filesystem.
        steps.append({'action': 'reuse_cudnn',
                      'detail': f'reuse existing cuDNN override at {cudnn_dir}'})
    else:
        if not discovery.get('writable_home'):
            return {'steps': [], 'blocked': 'Home directory is not writable.',
                    'af2_home': af2_home, 'cudnn_dir': None}
        cudnn_dir = f'{cudnn_target}/nvidia/cudnn/lib'
        steps.append({
            'action': 'install_cudnn',
            'detail': (f'download {CUDNN_PACKAGE} (~700 MB) to {cudnn_target} '
                       '— no shared copy found'),
        })

    current = discovery.get('wrapper_version')
    if current is None:
        steps.append({'action': 'write_wrapper',
                      'detail': f'create wrappers in {wrapper_dir}'})
    elif current < WRAPPER_VERSION:
        steps.append({
            'action': 'write_wrapper',
            'detail': (f'update wrappers in {wrapper_dir} '
                       f'(v{current} -> v{WRAPPER_VERSION})'),
        })

    return {'steps': steps, 'blocked': None,
            'af2_home': af2_home, 'cudnn_dir': cudnn_dir}


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


def _write_remote_file(conn: Connection, path: str, content: str,
                       executable: bool = False):
    """Write a file remotely without quoting hazards.

    The content is base64'd rather than heredoc'd: the wrapper contains
    single quotes, backslashes and `$@`, all of which are painful to pass
    through a shell safely. Base64's alphabet is inert inside single quotes.
    """
    payload = base64.b64encode(content.encode()).decode()
    cmd = f"printf '%s' '{payload}' | base64 -d > {path}"
    if executable:
        cmd += f" && chmod +x {path}"
    result = conn.run(cmd, hide=True, warn=True, timeout=60)
    if not result.ok:
        raise RuntimeError(f"could not write {path}: "
                           f"{(result.stderr or '').strip()[:200]}")


def apply_af2_setup(conn: Connection, plan: Dict,
                    wrapper_dir: str = "$HOME/bin",
                    cudnn_target: str = "$HOME/cudnn89",
                    on_step: Optional[Callable[[str], None]] = None
                    ) -> List[Dict]:
    """Execute a plan from plan_af2_setup. Returns one result per step.

    Every action is additive and confined to the user's home: nothing is
    deleted, no system path is touched, and sudo is never used. Failures are
    reported per step rather than raised, so a partial setup still explains
    what landed.
    """
    results = []

    def note(message: str):
        if on_step:
            on_step(message)

    for step in plan.get('steps', []):
        action = step['action']
        try:
            if action == 'reuse_cudnn':
                note(step['detail'])
                results.append({**step, 'ok': True, 'message': 'reused'})

            elif action == 'install_cudnn':
                note(step['detail'])
                py = f"{plan['af2_home']}/colabfold-conda/bin/python3.10"
                # --no-deps: the env already carries cuBLAS/cuFFT/cuSOLVER
                # for CUDA 11; only cuDNN is the wrong version.
                cmd = (f'PYTHONNOUSERSITE=1 "{py}" -m pip install '
                       f'--no-cache-dir --no-deps --target {cudnn_target} '
                       f'{CUDNN_PACKAGE}')
                result = conn.run(cmd, hide=True, warn=True, timeout=1800)
                results.append({**step, 'ok': result.ok,
                                'message': 'installed' if result.ok
                                else (result.stderr or '').strip()[:200]})

            elif action == 'write_wrapper':
                note(step['detail'])
                conn.run(f'mkdir -p {wrapper_dir}', hide=True, warn=True,
                         timeout=30)
                written = []
                for entry in WRAPPER_ENTRY_POINTS:
                    script = render_wrapper(plan['af2_home'],
                                            plan['cudnn_dir'], entry)
                    _write_remote_file(conn, f'{wrapper_dir}/{entry}', script,
                                       executable=True)
                    written.append(entry)
                results.append({**step, 'ok': True,
                                'message': f"wrote {', '.join(written)}"})
            else:
                results.append({**step, 'ok': False,
                                'message': f'unknown action {action}'})
        except Exception as exc:
            results.append({**step, 'ok': False, 'message': str(exc)[:200]})

    return results
