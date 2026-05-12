"""InterPLM: per-residue SAE features from ESM-2 embeddings.

Computes sparse autoencoder (SAE) feature activations for a single protein
sequence using the pre-trained models from Simon & Zou (2024).

HuggingFace weights: Elana/InterPLM-esm2-8m
ESM-2 model:        facebook/esm2_t6_8M_UR50D (6 layers, 320-dim)

torch, transformers, and huggingface_hub are installed automatically on
first use if missing (via pip). No manual setup required.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_CACHE_DIR = Path.home() / ".beak" / "interplm"
_CONCEPTS_DIR = Path.home() / ".beak" / "interplm" / "concepts"
_TOP_N = 200
_DEFAULT_MODEL = "8m"
_DEFAULT_LAYER = 6  # default layer for 8m

_MODELS: Dict[str, Dict] = {
    "8m": {
        "label":       "ESM-2-8M",
        "esm_id":      "facebook/esm2_t6_8M_UR50D",
        "sae_repo":    "Elana/InterPLM-esm2-8m",
        "n_layers":    6,
        "dim":         320,
        "layers":      [1, 2, 3, 4, 5, 6],
        "default_layer": 6,
    },
    "650m": {
        "label":       "ESM-2-650M",
        "esm_id":      "facebook/esm2_t33_650M_UR50D",
        "sae_repo":    "Elana/InterPLM-esm2-650m",
        "n_layers":    33,
        "dim":         1280,
        "layers":      [1, 9, 18, 24, 30, 33],
        "default_layer": 33,
    },
}

_REQUIRED_PKGS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
]

# Cached ESM-2 (tokenizer, model) keyed by model key ("8m", "650m").
# Avoids re-downloading on every layer switch and prevents tqdm's multiprocessing
# RLock from tripping "bad value(s) in fds_to_keep" inside Textual worker threads.
_esm_cache: Dict[str, tuple] = {}


def missing_deps() -> List[str]:
    """Return list of pip package names that aren't importable yet."""
    out = []
    for pip_name, import_name in _REQUIRED_PKGS:
        try:
            __import__(import_name)
        except ImportError:
            out.append(pip_name)
    return out


def ensure_deps(progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """Install any missing deps with pip, then make them importable.

    Python caches failed imports in sys.modules as None — leaving those
    stale entries in place blocks reimport even after a successful pip
    install in the same process. We clear them before invalidating the
    path cache so subsequent __import__ calls find the new packages.

    Raises RuntimeError if the pip install fails.
    """
    pkgs = missing_deps()
    if not pkgs:
        return

    if progress_cb:
        progress_cb(f"Installing {', '.join(pkgs)} (first run — may take a moment)…")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed for {pkgs}:\n{result.stderr.strip()}"
        )

    # Clear any cached None entries left by earlier failed imports.
    for _pip, import_name in _REQUIRED_PKGS:
        if sys.modules.get(import_name) is None:
            sys.modules.pop(import_name, None)
    importlib.invalidate_caches()


def compute_features(
    sequence: str,
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Compute InterPLM SAE features for a single protein sequence.

    Args:
        sequence: Amino acid sequence (1-letter codes, no gaps).
        layer:    Transformer layer to use. Valid layers depend on model.
        model:    Model key — "8m" or "650m".
        progress_cb: Optional callback receiving progress strings.

    Returns:
        dict with keys:
          feature_acts  np.ndarray (seq_len, n_features) float32
          top_features  list of dicts: {idx, max_act, mean_act, n_active}
          layer         int
          model         str
          seq_len       int
    """
    ensure_deps(progress_cb)

    cfg = _MODELS.get(model)
    if cfg is None:
        raise ValueError(f"Unknown model '{model}'. Valid: {list(_MODELS)}")
    if layer not in cfg["layers"]:
        raise ValueError(f"Layer {layer} not available for {model}. Valid: {cfg['layers']}")

    hidden = _get_esm2_embedding(sequence, layer, model, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(f"Loading InterPLM SAE ({cfg['label']} layer {layer})…")
    W_enc, b_enc, b_pre = _load_sae(layer, model, progress_cb=progress_cb)

    if progress_cb:
        progress_cb("Computing features…")
    acts = _sae_forward(hidden, W_enc, b_enc, b_pre)

    top = _rank_features(acts, n=_TOP_N)

    return {
        "feature_acts": acts,
        "top_features": top,
        "layer": layer,
        "model": model,
        "seq_len": len(sequence),
    }


def _patch_hf_progress() -> None:
    """Replace huggingface_hub's _get_progress_bar_context with a no-op.

    tqdm.__new__ always calls get_lock(), which spawns a multiprocessing
    resource-tracker subprocess. On macOS inside a Textual worker thread that
    subprocess inherits open FDs that it can't pass to its child, tripping
    "bad value(s) in fds_to_keep". Patching the context factory avoids
    creating any tqdm instance — and therefore any multiprocessing lock.
    """
    from contextlib import nullcontext

    class _NullProgress:
        def update(self, *a, **k):
            pass

    try:
        import huggingface_hub.file_download as _fd
        if not getattr(_fd, "_beak_patched", False):
            _fd._get_progress_bar_context = lambda **_kw: nullcontext(_NullProgress())
            _fd._beak_patched = True
    except Exception:
        pass


def _get_esm2_embedding(
    sequence: str,
    layer: int,
    model: str = _DEFAULT_MODEL,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Run ESM-2 on sequence; return hidden states at `layer`. Shape: (L, dim)."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    _patch_hf_progress()

    cfg = _MODELS[model]
    esm_id = cfg["esm_id"]
    if model not in _esm_cache:
        if progress_cb:
            progress_cb(f"Downloading {cfg['label']} tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(esm_id)
        if progress_cb:
            progress_cb(f"Downloading {cfg['label']} weights…")
        # `low_cpu_mem_usage=False` + explicit `device_map=None` + post-load
        # `.to("cpu")` is the belt-and-braces fix for the
        # ``Tensor on device meta is not on the expected device cpu!``
        # error transformers raises when ``accelerate`` is installed and
        # silently switches to meta-device lazy init for the 650M model.
        # The .to() call materialises any lingering meta tensors onto CPU.
        esm = EsmModel.from_pretrained(
            esm_id,
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=torch.float32,
        ).to("cpu")
        esm.eval()
        _esm_cache[model] = (tokenizer, esm)
        if progress_cb:
            progress_cb(f"Running {cfg['label']}…")
    else:
        if progress_cb:
            progress_cb(f"Running {cfg['label']}…")
    tokenizer, esm = _esm_cache[model]

    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = esm(**inputs, output_hidden_states=True)

    # hidden_states[0] = embedding layer; [1..N] = transformer blocks
    hidden = outputs.hidden_states[layer]  # (1, L+2, dim)
    hidden = hidden[0, 1:-1].float().numpy()  # strip BOS/EOS
    return hidden.astype(np.float32)


def _load_sae(
    layer: int,
    model: str = _DEFAULT_MODEL,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download (once) and load SAE weights for the given model+layer.

    Bypasses ``huggingface_hub`` — its tqdm progress bar eagerly constructs
    a multiprocessing RLock that trips ``bad value(s) in fds_to_keep`` inside
    a Textual worker thread on macOS.
    """
    import torch
    import urllib.request

    cfg = _MODELS[model]
    sae_cache = _CACHE_DIR / f"sae_{model}_layer_{layer}.pt"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate old 8m cache written before multi-model support.
    if not sae_cache.exists() and model == "8m":
        old = _CACHE_DIR / f"sae_layer_{layer}.pt"
        if old.exists():
            old.rename(sae_cache)

    if not sae_cache.exists():
        url = (
            f"https://huggingface.co/{cfg['sae_repo']}/resolve/main/"
            f"layer_{layer}/ae_normalized.pt"
        )
        tmp = sae_cache.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb:
                            mb = done / (1024 * 1024)
                            if total:
                                pct = int(done / total * 100)
                                tot_mb = total / (1024 * 1024)
                                progress_cb(
                                    f"Downloading {cfg['label']} SAE layer {layer}… "
                                    f"{mb:.0f}/{tot_mb:.0f} MB ({pct}%)"
                                )
                            else:
                                progress_cb(
                                    f"Downloading {cfg['label']} SAE layer {layer}… "
                                    f"{mb:.0f} MB"
                                )
            tmp.replace(sae_cache)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    state = torch.load(sae_cache, map_location="cpu", weights_only=True)
    W_enc = state["encoder.weight"].float().numpy()  # (n_features, d_model)
    b_enc = state["encoder.bias"].float().numpy()    # (n_features,)
    b_pre = state["bias"].float().numpy()            # (d_model,)
    return W_enc, b_enc, b_pre


def _sae_forward(
    hidden: np.ndarray,
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    b_pre: np.ndarray,
) -> np.ndarray:
    """Encode hidden states through the SAE.

    InterPLM's SAE forward: f(x) = ReLU(W_enc @ (x - b_pre) + b_enc).
    Returns float32 array of shape (seq_len, n_features).
    """
    centered = hidden - b_pre[np.newaxis, :]
    pre = centered @ W_enc.T + b_enc[np.newaxis, :]
    return np.maximum(0.0, pre, dtype=np.float32)


# ── Cache helpers ─────────────────────────────────────────────────────────


def cache_path(
    project_path: Path,
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
) -> Path:
    """On-disk path for cached feature activations."""
    return Path(project_path) / "target" / "interplm" / model / f"layer_{layer}.npz"


def is_cached(
    project_path: Path,
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
) -> bool:
    p = cache_path(project_path, layer, model)
    if p.exists():
        return True
    # Migrate flat-layout caches written before multi-model support.
    if model == "8m":
        old = Path(project_path) / "target" / "interplm" / f"layer_{layer}.npz"
        if old.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            old.rename(p)
            return True
    return False


def save_to_cache(project_path: Path, result: dict) -> None:
    """Persist feature_acts as float16 npz (saves ~50% vs float32)."""
    p = cache_path(project_path, result["layer"], result.get("model", _DEFAULT_MODEL))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        feature_acts=result["feature_acts"].astype(np.float16),
        layer=np.array(result["layer"]),
        seq_len=np.array(result["seq_len"]),
    )
    tmp.replace(p)


def load_from_cache(
    project_path: Path,
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
) -> dict:
    """Load cached feature activations and re-derive top_features."""
    p = cache_path(project_path, layer, model)
    with np.load(p) as data:
        acts = data["feature_acts"].astype(np.float32)
        seq_len = int(data["seq_len"])
    return {
        "feature_acts": acts,
        "top_features": _rank_features(acts),
        "layer": layer,
        "model": model,
        "seq_len": seq_len,
    }


# ── Concept annotations (one-time remote computation) ─────────────────────


def concepts_path(layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL) -> Path:
    return _CONCEPTS_DIR / model / f"layer_{layer}.csv"


def concepts_meta_path(layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL) -> Path:
    """Sidecar JSON next to the CSV: {model, layer, n_proteins, n_features, computed_at}."""
    return _CONCEPTS_DIR / model / f"layer_{layer}.meta.json"


def has_concepts(layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL) -> bool:
    return concepts_path(layer, model).exists()


def load_concepts_meta(
    layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL,
) -> Optional[Dict]:
    """Return sidecar metadata or None when missing/unreadable."""
    import json as _json
    p = concepts_meta_path(layer, model)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return _json.load(f)
    except Exception:
        return None


def delete_concepts(layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL) -> None:
    """Remove the concepts CSV and its sidecar metadata if present."""
    for p in (concepts_path(layer, model), concepts_meta_path(layer, model)):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# ── Active-job tracking (process-global, for UI surfacing) ────────────────
#
# `compute_concepts_remote` is called from a Textual worker thread on the
# InterPLM screen. If the user navigates away, the worker keeps running but
# the screen instance is gone. The layers panel (on project detail) needs a
# way to see "concepts running" without owning the worker. A module-level
# dict keyed on (model, layer) is the simplest correct primitive — the
# worker writes here on each progress callback, any reader can poll it.

import threading as _threading
import time as _time

_concepts_jobs_lock = _threading.Lock()
_concepts_jobs: Dict[Tuple[str, int], Dict] = {}


def _set_concepts_progress(model: str, layer: int, message: str) -> None:
    with _concepts_jobs_lock:
        rec = _concepts_jobs.setdefault((model, layer), {
            "model": model, "layer": layer, "started_at": _time.time(),
        })
        rec["status"] = "running"
        rec["message"] = message
        rec["updated_at"] = _time.time()


def _finish_concepts_progress(
    model: str, layer: int, status: str, message: str = "",
) -> None:
    """Mark the job done/failed; UI can show a brief terminal state then clear."""
    with _concepts_jobs_lock:
        rec = _concepts_jobs.get((model, layer))
        if rec is None:
            rec = {"model": model, "layer": layer, "started_at": _time.time()}
            _concepts_jobs[(model, layer)] = rec
        rec["status"] = status
        rec["message"] = message
        rec["updated_at"] = _time.time()


def _clear_concepts_progress(model: str, layer: int) -> None:
    with _concepts_jobs_lock:
        _concepts_jobs.pop((model, layer), None)


def get_concepts_progress(
    model: str, layer: int,
) -> Optional[Dict]:
    """Return current state for one (model, layer), or None if no job."""
    with _concepts_jobs_lock:
        rec = _concepts_jobs.get((model, layer))
        return dict(rec) if rec else None


def get_active_concepts_jobs() -> Dict[Tuple[str, int], Dict]:
    """Return a snapshot of every (model, layer) with a known job state."""
    with _concepts_jobs_lock:
        return {k: dict(v) for k, v in _concepts_jobs.items()}


def load_concepts(layer: int = _DEFAULT_LAYER, model: str = _DEFAULT_MODEL) -> Dict[int, str]:
    """Load concepts CSV → {feature_idx: concept_name}."""
    import csv as _csv
    result: Dict[int, str] = {}
    p = concepts_path(layer, model)
    if not p.exists():
        return result
    with open(p, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                result[int(row["feature"])] = row["concept"]
            except (KeyError, ValueError):
                pass
    return result


def _write_concepts_error_log(
    model: str,
    layer: int,
    n_proteins: int,
    streamer: Optional["_LineStreamer"],
    result,
    exc: Optional[BaseException],
) -> Path:
    """Persist the full failure context to ``~/.beak/errors/`` and return the path.

    The on-screen error toast can only show ~80 chars before wrapping;
    real remote tracebacks (especially CUDA / transformers kernels) need
    a full log to diagnose. We dump everything we have: streamed
    stdout/stderr, the remote return code, and the local exception
    traceback. Caller surfaces the path in the toast so the user can
    open the log in their editor.
    """
    import traceback as _tb
    from datetime import datetime as _dt

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / ".beak" / "errors"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ts}_concepts_{model}_layer_{layer}.log"

    with open(log_path, "w") as f:
        f.write(f"# concept compute failure · {ts}\n")
        f.write(f"# model={model}  layer={layer}  n_proteins={n_proteins}\n\n")
        if result is not None:
            f.write(f"# remote return_code: {getattr(result, 'return_code', '?')}\n\n")
        if exc is not None:
            f.write(f"# local exception: {type(exc).__name__}: {exc}\n\n")
            f.write("--- local traceback ---\n")
            _tb.print_exception(type(exc), exc, exc.__traceback__, file=f)
            f.write("\n")
        if streamer is not None and streamer.lines:
            f.write("--- remote streamed output ---\n")
            for line in streamer.lines:
                f.write(line + "\n")
        elif result is not None:
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            if stdout:
                f.write("--- remote stdout ---\n")
                f.write(stdout)
                f.write("\n")
            if stderr:
                f.write("--- remote stderr ---\n")
                f.write(stderr)
                f.write("\n")
    return log_path


class _LineStreamer:
    """File-like object that fires a callback for each complete line received.

    Used as Fabric's ``out_stream`` so remote script print() calls are relayed
    to the TUI loading label in real time instead of appearing all at once.
    """

    def __init__(self, cb: Callable[[str], None]) -> None:
        self._cb = cb
        self._buf = ""
        self.lines: list = []

    def write(self, data: str) -> int:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.lines.append(line)
            stripped = line.strip()
            if stripped:
                self._cb(stripped)
        return len(data)

    def flush(self) -> None:
        pass


def compute_concepts_remote(
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
    n_proteins: int = 10000,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Run SAE concept computation on the remote server.

    Uploads ``remote/interplm_concepts.py`` (a real file in the package,
    not a templated string), runs it on the remote with CLI args, and
    downloads the resulting CSV to
    ``~/.beak/interplm/concepts/<model>/layer_<layer>.csv``.

    For a smoke test, pass ``n_proteins=100`` (~2 min on remote GPU).
    The default 10,000 is the production setting (~30–40 min on GPU).

    Returns the number of features annotated.
    Requires SSH access configured via ``beak config init``.
    """
    from fabric import Connection as FabricConnection
    from .config import get_default_connection

    def _cb(msg: str) -> None:
        # Mirror every progress message into the module-level tracker so
        # other UI surfaces (layers panel, a re-mounted InterPLM screen)
        # can read it without owning the worker.
        _set_concepts_progress(model, layer, msg)
        if progress_cb:
            progress_cb(msg)

    _cb("Connecting to remote…")
    defaults = get_default_connection()
    if not defaults.get("host"):
        _finish_concepts_progress(model, layer, "failed", "no remote configured")
        raise RuntimeError("No remote configured — run: beak config init")

    ckw: dict = {}
    if defaults.get("key_path"):
        ckw["key_filename"] = defaults["key_path"]

    conn = FabricConnection(
        host=defaults["host"],
        user=defaults.get("user", ""),
        connect_kwargs=ckw,
    )
    try:
        cfg = _MODELS[model]
        slug = f"{model}_layer{layer}"
        remote_script = f"/tmp/beak_interplm_concepts_{slug}.py"
        remote_csv = f"/tmp/beak_interplm_concepts_{slug}.csv"

        script_src = Path(__file__).parent / "remote" / "interplm_concepts.py"
        if not script_src.exists():
            raise RuntimeError(
                f"Concept script missing at {script_src} — package install broken?"
            )

        _cb("Uploading concept script…")
        conn.put(local=str(script_src), remote=remote_script)

        runtime_hint = "~2 min" if n_proteins <= 200 else (
            "~10 min" if n_proteins <= 2000 else "~30–40 min"
        )
        _cb(
            f"Running on remote — {cfg['label']} layer {layer}, "
            f"{n_proteins} proteins ({runtime_hint})…"
        )
        cmd = (
            f"python3 {remote_script} "
            f"--model {model} --layer {layer} "
            f"--n-proteins {n_proteins} "
            f"--output {remote_csv} 2>&1"
        )
        streamer = _LineStreamer(_cb) if progress_cb else _LineStreamer(lambda _: None)
        # `in_stream=False` is critical: Fabric defaults to piping local
        # sys.stdin into the remote process, but inside Textual stdin is
        # captured for keypress handling. When the remote exits, Fabric's
        # stdin pump fires one last write to the (now-closed) SSH channel
        # and raises ``OSError: Socket is closed`` — making us drop the
        # result of an otherwise-successful remote run.
        result = conn.run(
            cmd, hide=True, warn=True, in_stream=False, out_stream=streamer,
        )
        if result.return_code != 0:
            log_path = _write_concepts_error_log(
                model, layer, n_proteins, streamer, result, None,
            )
            tail = "\n".join(streamer.lines[-12:])
            raise RuntimeError(
                f"Remote script failed (exit {result.return_code}). "
                f"Full log: {log_path}\n\nLast lines:\n{tail}"
            )

        _cb("Downloading concepts CSV…")
        local_csv = concepts_path(layer, model)
        local_csv.parent.mkdir(parents=True, exist_ok=True)
        conn.get(remote=remote_csv, local=str(local_csv))
        conn.run(f"rm -f {remote_script} {remote_csv}", hide=True, warn=True)

        with open(local_csv) as f:
            n_annotated = sum(1 for _ in f) - 1

        # Sidecar metadata for the KB management view — captures what
        # was actually run so a future "this looks stale, recompute?"
        # decision is grounded in real provenance, not just file mtime.
        import json as _json
        from datetime import datetime, timezone
        meta = {
            "model": model,
            "layer": layer,
            "n_proteins_requested": n_proteins,
            "n_features_annotated": n_annotated,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "methodology_version": 2,  # asymmetric F1 + threshold sweep
            "f1_min": 0.4,
            "thresholds": [0.15, 0.5, 0.8],
        }
        meta_path = concepts_meta_path(layer, model)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=2)

        _cb(f"Done — {n_annotated} features annotated.")
        _finish_concepts_progress(
            model, layer, "done",
            f"{n_annotated} features annotated",
        )
        return n_annotated
    except Exception as e:
        # If `e` already references a log path (raised from the
        # return-code branch), don't write a duplicate. Otherwise this
        # is an unexpected exception (SSH drop, local manifest write
        # failure, etc.) — capture it for post-mortem.
        emsg = str(e)
        if "Full log:" not in emsg:
            log_path = _write_concepts_error_log(
                model, layer, n_proteins,
                locals().get("streamer"),
                locals().get("result"),
                e,
            )
            _finish_concepts_progress(
                model, layer, "failed",
                f"{type(e).__name__}: see {log_path.name}",
            )
            raise RuntimeError(
                f"{type(e).__name__}: {emsg}\n\nFull log: {log_path}"
            ) from e
        _finish_concepts_progress(model, layer, "failed", emsg[:120])
        raise
    finally:
        conn.close()


_SW_TOP_N = 100


def swissprot_top100_path(layer: int, model: str = _DEFAULT_MODEL) -> Path:
    cfg = _MODELS[model]
    slug = cfg["label"].lower().replace("-", "_").replace(" ", "_")
    return _CACHE_DIR / f"swissprot_top100_{slug}_layer_{layer}.npz"


def has_swissprot_top100(layer: int, model: str = _DEFAULT_MODEL) -> bool:
    return swissprot_top100_path(layer, model).exists()


def load_swissprot_top100(
    layer: int, model: str = _DEFAULT_MODEL
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top_ids, top_acts) for precomputed Swiss-Prot top-100.

    top_ids:  (n_features, TOP_N) dtype U10  — UniProt accessions, rank-0 highest
    top_acts: (n_features, TOP_N) dtype f16  — corresponding max activations
    """
    p = swissprot_top100_path(layer, model)
    data = np.load(p)
    return data["top_ids"], data["top_acts"]


def compute_swissprot_remote(
    layer: int = _DEFAULT_LAYER,
    model: str = _DEFAULT_MODEL,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Precompute Swiss-Prot top-100 proteins per SAE feature on the remote server.

    Uploads a self-contained Python script, runs it on the remote (GPU
    recommended), downloads the result to
    ~/.beak/interplm/swissprot_top100_{model}_layer_{layer}.npz.
    """
    import os
    import tempfile
    from fabric import Connection as FabricConnection
    from .config import get_default_connection

    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _cb("Connecting to remote…")
    defaults = get_default_connection()
    if not defaults.get("host"):
        raise RuntimeError("No remote configured — run: beak config init")

    ckw: dict = {}
    if defaults.get("key_path"):
        ckw["key_filename"] = defaults["key_path"]

    conn = FabricConnection(
        host=defaults["host"],
        user=defaults.get("user", ""),
        connect_kwargs=ckw,
    )
    try:
        cfg = _MODELS[model]
        slug = f"{model}_layer{layer}"
        remote_script = f"/tmp/beak_interplm_swissprot_{slug}.py"
        remote_npz = f"/tmp/beak_interplm_swissprot_{slug}.npz"

        local_out = swissprot_top100_path(layer, model)
        script = _make_swissprot_script(layer, cfg["esm_id"], cfg["sae_repo"], remote_npz)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write(script)
            tmp_path = tf.name

        _cb("Uploading script…")
        conn.put(local=tmp_path, remote=remote_script)
        os.unlink(tmp_path)

        _cb(
            f"Running on remote — {cfg['label']} layer {layer}, "
            "~570K Swiss-Prot proteins (~20 min on GPU)…"
        )
        streamer = _LineStreamer(_cb) if progress_cb else None
        result = conn.run(
            f"python3 {remote_script} 2>&1",
            hide=True,
            warn=True,
            in_stream=False,  # see compute_concepts_remote for rationale
            out_stream=streamer,
        )
        if result.return_code != 0:
            captured = streamer.lines if streamer else result.stdout.strip().splitlines()
            tail = "\n".join(captured[-8:])
            raise RuntimeError(f"Remote script failed:\n{tail}")

        _cb("Downloading result npz…")
        local_out.parent.mkdir(parents=True, exist_ok=True)
        conn.get(remote=remote_npz, local=str(local_out))
        conn.run(f"rm -f {remote_script} {remote_npz}", hide=True, warn=True)

        _cb("Done — Swiss-Prot top-100 precompute complete.")
    finally:
        conn.close()


def _make_swissprot_script(
    layer: int, esm_id: str, sae_repo: str, output_path: str
) -> str:
    """Return a self-contained Python script for Swiss-Prot top-100 precompute."""
    body = (
        '"""InterPLM SAE Swiss-Prot top-100 — auto-generated by beak."""\n'
        "import sys, os, gzip, heapq, subprocess\n"
        "import urllib.request\n\n"
        "LAYER   = __LAYER__\n"
        "TOP_N   = 100\n"
        "OUTPUT  = __OUTPUT__\n"
        f"SAE_REPO = {repr(sae_repo)}\n"
        f"ESM2_ID  = {repr(esm_id)}\n"
        "FASTA_URL = (\n"
        "    'https://ftp.uniprot.org/pub/databases/uniprot/'\n"
        "    'current_release/knowledgebase/complete/uniprot_sprot.fasta.gz'\n"
        ")\n"
        "FASTA_CACHE = '/tmp/uniprot_sprot.fasta.gz'\n\n"
        "for _pkg in ['torch', 'transformers']:\n"
        "    try: __import__(_pkg)\n"
        "    except ImportError:\n"
        "        print(f'Installing {_pkg}...', flush=True)\n"
        "        subprocess.run([sys.executable,'-m','pip','install','-q',_pkg], check=True)\n"
        "import importlib; importlib.invalidate_caches()\n\n"
        "import torch, numpy as np\n"
        "from transformers import AutoTokenizer, EsmModel\n\n"
        "if not os.path.exists(FASTA_CACHE):\n"
        "    print('Downloading Swiss-Prot FASTA (~85 MB)...', flush=True)\n"
        "    with urllib.request.urlopen(FASTA_URL, timeout=600) as _r:\n"
        "        with open(FASTA_CACHE+'.tmp', 'wb') as _f:\n"
        "            while True:\n"
        "                chunk = _r.read(1 << 20)\n"
        "                if not chunk: break\n"
        "                _f.write(chunk)\n"
        "    os.rename(FASTA_CACHE+'.tmp', FASTA_CACHE)\n"
        "    print('Downloaded.', flush=True)\n\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print(f'Device: {device}', flush=True)\n"
        "print(f'Loading {ESM2_ID}...', flush=True)\n"
        "tokenizer = AutoTokenizer.from_pretrained(ESM2_ID)\n"
        "esm = EsmModel.from_pretrained(ESM2_ID); esm.eval().to(device)\n\n"
        "sae_cache = f'/tmp/beak_sae_layer{LAYER}.pt'\n"
        "if not os.path.exists(sae_cache):\n"
        "    sae_url = f'https://huggingface.co/{SAE_REPO}/resolve/main/layer_{LAYER}/ae_normalized.pt'\n"
        "    print(f'Downloading SAE layer {LAYER}...', flush=True)\n"
        "    with urllib.request.urlopen(sae_url, timeout=300) as _r:\n"
        "        with open(sae_cache+'.tmp', 'wb') as _f: _f.write(_r.read())\n"
        "    os.rename(sae_cache+'.tmp', sae_cache)\n"
        "state = torch.load(sae_cache, map_location='cpu', weights_only=True)\n"
        "W_enc = torch.tensor(state['encoder.weight'].float()).to(device)\n"
        "b_enc = torch.tensor(state['encoder.bias'].float()).to(device)\n"
        "b_pre = torch.tensor(state['bias'].float()).to(device)\n"
        "n_feat = W_enc.shape[0]\n"
        "print(f'SAE: {n_feat} features', flush=True)\n\n"
        "heaps = [[] for _ in range(n_feat)]\n\n"
        "def _update(accession, max_acts):\n"
        "    active = np.where(max_acts > 0.0)[0]\n"
        "    for fi in active:\n"
        "        v = float(max_acts[fi])\n"
        "        hp = heaps[fi]\n"
        "        if len(hp) < TOP_N:\n"
        "            heapq.heappush(hp, (v, accession))\n"
        "        elif v > hp[0][0]:\n"
        "            heapq.heapreplace(hp, (v, accession))\n\n"
        "n_proc = 0\n"
        "cur_acc = ''\n"
        "seq_buf = []\n"
        "with gzip.open(FASTA_CACHE, 'rt', encoding='utf-8', errors='replace') as fh:\n"
        "    for line in fh:\n"
        "        line = line.rstrip('\\n')\n"
        "        if line.startswith('>'):\n"
        "            if cur_acc and seq_buf:\n"
        "                seq = ''.join(seq_buf)\n"
        "                if 20 <= len(seq) <= 1022:\n"
        "                    try:\n"
        "                        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1022)\n"
        "                        inp = {k: v.to(device) for k,v in inp.items()}\n"
        "                        with torch.no_grad():\n"
        "                            hid = esm(**inp, output_hidden_states=True).hidden_states[LAYER][0,1:-1]\n"
        "                        acts = torch.relu((hid - b_pre) @ W_enc.T + b_enc).max(0).values\n"
        "                        _update(cur_acc, acts.cpu().float().numpy())\n"
        "                    except Exception: pass\n"
        "                n_proc += 1\n"
        "                if n_proc % 10000 == 0: print(f'Processed {n_proc}...', flush=True)\n"
        "            parts = line.split('|')\n"
        "            cur_acc = parts[1] if len(parts) >= 2 else line[1:11]\n"
        "            seq_buf = []\n"
        "        else:\n"
        "            seq_buf.append(line)\n"
        "    if cur_acc and seq_buf:\n"
        "        seq = ''.join(seq_buf)\n"
        "        if 20 <= len(seq) <= 1022:\n"
        "            try:\n"
        "                inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1022)\n"
        "                inp = {k: v.to(device) for k,v in inp.items()}\n"
        "                with torch.no_grad():\n"
        "                    hid = esm(**inp, output_hidden_states=True).hidden_states[LAYER][0,1:-1]\n"
        "                acts = torch.relu((hid - b_pre) @ W_enc.T + b_enc).max(0).values\n"
        "                _update(cur_acc, acts.cpu().float().numpy())\n"
        "            except Exception: pass\n"
        "        n_proc += 1\n\n"
        "print(f'Processed {n_proc} total sequences', flush=True)\n"
        "print('Building output arrays...', flush=True)\n"
        "top_ids  = np.full((n_feat, TOP_N), '', dtype='U10')\n"
        "top_acts_arr = np.zeros((n_feat, TOP_N), dtype=np.float16)\n"
        "for fi in range(n_feat):\n"
        "    for rank, (v, aid) in enumerate(sorted(heaps[fi], reverse=True)):\n"
        "        top_ids[fi, rank] = aid\n"
        "        top_acts_arr[fi, rank] = np.float16(v)\n"
        "os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)) or '.', exist_ok=True)\n"
        "np.savez_compressed(OUTPUT, top_ids=top_ids, top_acts=top_acts_arr)\n"
        "print(f'Done — {n_feat} features x {TOP_N} proteins saved to {OUTPUT}', flush=True)\n"
    )
    return (
        "#!/usr/bin/env python3\n"
        + body
        .replace("__LAYER__", str(layer))
        .replace("__OUTPUT__", repr(output_path))
    )


def _rank_features(acts: np.ndarray, n: int = _TOP_N) -> List[Dict]:
    """Return top-N features sorted descending by max activation across residues."""
    # acts: (L, n_features)
    max_acts = acts.max(axis=0)        # (n_features,)
    mean_acts = acts.mean(axis=0)      # (n_features,)
    threshold = 0.1
    n_active = (acts > threshold).sum(axis=0)  # (n_features,)

    # Argsort descending by max_act
    order = np.argsort(max_acts)[::-1][:n]
    result = []
    for idx in order:
        if max_acts[idx] <= 0.0:
            break
        result.append({
            "idx": int(idx),
            "max_act": float(max_acts[idx]),
            "mean_act": float(mean_acts[idx]),
            "n_active": int(n_active[idx]),
        })
    return result
