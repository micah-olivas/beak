#!/usr/bin/env python3
"""ESM embedding generation script to run inside the Docker container.

Design notes:
  - Each sequence is embedded independently and its result is written to an
    .npz chunk *before* moving on. This means a partial failure (OOM, bad
    sequence, network blip killing the pod) loses at most one sequence.
  - On restart the script scans the chunks directory and skips any seq_id
    that already has an output chunk — so rerunning the same run.sh resumes.
  - A progress.json beside the chunks tracks {done, total, current,
    started_at, last_update, failed} and is updated every sequence. Tail it
    remotely to see real-time progress; the log file (stdout, flushed) shows
    per-sequence timings.
  - A final consolidation step stitches chunks into the mean_embeddings.pkl /
    per_token_embeddings.pkl files the Python loader expects, preserving the
    existing output contract.
"""

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MEAN_PICKLE = "mean_embeddings.pkl"
PER_TOK_PICKLE = "per_token_embeddings.pkl"
PROGRESS_FILE = "progress.json"
FAILED_FILE = "failed.tsv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_numpy(rep) -> np.ndarray:
    """Move a torch tensor (or numpy-already) to a float32 numpy array.

    ESMC's SDK returns hidden states in bfloat16 on GPU, and numpy has
    no native bfloat16 dtype — calling ``.numpy()`` on a bf16 tensor
    raises ``TypeError: Got unsupported ScalarType BFloat16``. We must
    cast to float32 (or float16) on the torch side first. Float32 is
    the safe target: it round-trips bfloat16 losslessly into the
    representable exponent range and downstream PCA / mean / cosine
    code expects float32 anyway.
    """
    if hasattr(rep, "cpu"):
        # `.float()` upcasts bfloat16/float16 → float32 on the GPU side
        # before the host transfer. Idempotent if the tensor's already
        # float32.
        rep = rep.detach().cpu().float().numpy()
    return np.asarray(rep)


def _to_mean_vector(rep) -> np.ndarray:
    """Mean-pool any rank-≥1 tensor to a 1-D `(D,)` numpy array.

    `rep` may be a torch tensor or already a numpy array. We convert
    via ``_to_numpy`` (which handles the bf16 → fp32 cast ESMC needs),
    then average every leading axis so the result is always 1-D
    regardless of whether the backend handed us `(seq_len, D)`,
    `(batch, seq_len, D)`, or something stranger. The last axis is
    the model's embedding dim, kept intact.
    """
    arr = _to_numpy(rep)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.astype(np.float32, copy=False)


def _save_chunk(path: Path, layers: dict) -> None:
    """Atomic-ish write: save to a sibling .tmp.npz, then rename over target.

    np.savez_compressed auto-appends .npz if the target doesn't already end
    in .npz, so we name the temp with the .npz suffix it expects and rename
    over the final path when the write completes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **layers)
    tmp.replace(path)


def _load_chunk(path: Path) -> dict:
    with np.load(path) as npz:
        return {name: npz[name] for name in npz.files}


def _write_progress(progress_path: Path, state: dict) -> None:
    tmp = progress_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(progress_path)


def _record_failure(failed_path: Path, seq_id: str, exc: BaseException) -> None:
    line = f"{seq_id}\t{type(exc).__name__}\t{str(exc).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
    with open(failed_path, 'a') as f:
        f.write(line)


def _consolidate_chunks(chunks_dir: Path, out_pickle: Path) -> int:
    """Merge all per-sequence .npz chunks under chunks_dir into a pickle dict."""
    merged = {}
    for chunk_path in sorted(chunks_dir.glob('*.npz')):
        seq_id = chunk_path.stem
        merged[seq_id] = _load_chunk(chunk_path)
    with open(out_pickle, 'wb') as f:
        pickle.dump(merged, f)
    return len(merged)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
# Each backend knows how to: load a model, report its layer count, and embed
# one sequence (returning a dict of {f"layer_N": np.ndarray(seq_len, D)}).
# The consolidation / chunking / progress logic below is backend-agnostic.
# ---------------------------------------------------------------------------

ESM_MAX_AA = 1022  # both ESM2 and ESM-C publicly trained on up to 1024 tokens


class Esm2Backend:
    """Loads ESM2 via `fair-esm`'s native API (same as before)."""

    name = 'esm2'

    def __init__(self, model_name: str, gpu_id: int):
        import esm as _esm
        import torch

        self.model, self.alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
        self.device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self._torch = torch

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    def normalize_layers(self, repr_layers: list) -> list:
        """Resolve negative layer indices for the ESM-2 indexing scheme.

        ESM-2's `out["representations"]` is a dict keyed by integer
        layer numbers 0..num_layers (inclusive of both ends — `0` is
        the input embedding, `num_layers` is after the final block).
        So `L = -1` should resolve to `num_layers`, not `num_layers-1`.
        """
        n = self.num_layers
        return [(n + L + 1) if L < 0 else L for L in repr_layers]

    def embed_one(self, seq_id: str, seq: str, repr_layers: list) -> dict:
        _, batch_strs, batch_tokens = self.batch_converter([(seq_id, seq)])
        batch_tokens = batch_tokens.to(self.device)
        with self._torch.no_grad():
            out = self.model(batch_tokens, repr_layers=repr_layers, return_contacts=False)

        seq_len = len(batch_strs[0])
        return {
            f"layer_{L}": out["representations"][L][0, 1:seq_len + 1]
            for L in repr_layers
        }


class EsmCBackend:
    """Loads ESM-C via EvolutionaryScale's `esm` SDK.

    Earlier versions of this file tried `transformers.AutoModel` with
    `trust_remote_code=True` to dodge a PyPI name-collision with
    fair-esm (both publish a top-level `esm` import). That path never
    worked: the EvolutionaryScale repos on HuggingFace ship the raw
    weights only, not transformers-compatible modeling code, so the
    AutoModel call failed for everyone who tried it.

    The supported loader is the SDK itself — installed in a separate
    venv inside the container (`/opt/esmc-venv`) so it can coexist
    with fair-esm in the system Python without name conflict. This
    backend is invoked from THAT venv (see `_make_backend` dispatch in
    `_generate_job_script`), not from the system Python.
    """

    name = 'esmc'

    # Short aliases -> SDK model identifiers.
    SDK_IDS = {
        'esmc_300m': 'esmc_300m',
        'esmc_600m': 'esmc_600m',
    }

    def __init__(self, model_name: str, gpu_id: int):
        import torch

        self.device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        )

        # FlashAttention + memory-efficient SDPA require Ampere
        # (SM 8.0+). On Pascal (1080 Ti, P100), Volta (V100), and
        # Turing (T4 / RTX 20-series for FA-2), the kernels either
        # refuse to dispatch or — worse — the `flash_attn` PyPI
        # package crashes at import. ESMC's SDK calls into
        # `torch.nn.functional.scaled_dot_product_attention`, which
        # does NOT auto-fall-back gracefully for every backend
        # version. Explicitly disable flash + mem-efficient on pre-
        # Ampere so PyTorch routes through the always-available
        # math kernel. Slower per token, but correct (and the only
        # option on the user's actual hardware).
        #
        # NOTE: mem-efficient SDPA *might* work on Pascal (cutlass
        # supports SM 5.0+) and would be 2-3× faster than math —
        # but enabling it speculatively has burned us before, and
        # the ESMC SDK can interact with the backend toggles in
        # surprising ways. If you want to try it, set the env var
        # `BEAK_ENABLE_MEM_EFFICIENT_SDP=1` at submit time and we'll
        # leave mem_efficient enabled. With it unset (the default),
        # we keep the conservative math-only path that's known to
        # work end-to-end on the user's 1080 Ti hardware.
        if self.device.type == 'cuda':
            try:
                cc = torch.cuda.get_device_capability(self.device)
            except Exception:
                cc = (0, 0)
            if cc < (8, 0):
                import os as _os
                opt_in = _os.environ.get(
                    'BEAK_ENABLE_MEM_EFFICIENT_SDP', '0'
                ) == '1'
                _log(
                    f"GPU compute capability {cc[0]}.{cc[1]} is "
                    f"pre-Ampere — pinning SDPA to "
                    f"{'mem-efficient + math (opt-in)' if opt_in else 'the math kernel'} "
                    f"(disabling flash"
                    f"{'' if opt_in else ' + mem-efficient'} backends)."
                )
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(opt_in)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    # Older torches don't expose these toggles —
                    # the in-built selection logic should still pick
                    # math when flash isn't available, just less
                    # reliably. Don't fail the job over it.
                    pass

        sdk_id = self.SDK_IDS.get(model_name, model_name)
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as e:
            # Distinguish the "wrong interpreter" case (no `esm` at
            # all) from the "FlashAttention import failed" case (esm
            # imports a sub-module that requires SM 7.5+). The latter
            # is what bites Pascal users on first run; surface the
            # specific cause instead of leaving them to dig into the
            # traceback.
            err_str = str(e).lower()
            if 'flash_attn' in err_str:
                raise RuntimeError(
                    "ESM-C's SDK pulled in `flash_attn` and the "
                    "import failed on this GPU. The `flash_attn` "
                    "PyPI wheel requires CUDA compute capability "
                    "≥ 7.5 (Turing). Your card appears to be older. "
                    "Workarounds: rebuild the embeddings container "
                    "with FA disabled, OR upgrade to an Ampere+ GPU "
                    "(see beak's submit-embed modal for the warning "
                    "tier). "
                    f"Original ImportError: {e}"
                ) from e
            raise RuntimeError(
                "ESM-C requires EvolutionaryScale's `esm` SDK. The "
                "container should have it pre-installed at "
                "/opt/esmc-venv; if you're running this script from "
                "the system Python that's the bug — dispatch must "
                "use the venv interpreter for esmc_* models. "
                f"Original ImportError: {e}"
            ) from e

        try:
            self.model = ESMC.from_pretrained(sdk_id).to(self.device).eval()
        except (RuntimeError, ImportError) as e:
            # `from_pretrained` may import `flash_attn` lazily (the
            # SDK does this on some versions). Same fail-fast path.
            err_str = str(e).lower()
            if 'flash' in err_str or 'sm' in err_str or 'capability' in err_str:
                raise RuntimeError(
                    "ESM-C model load failed, likely because of a "
                    "FlashAttention / compute-capability mismatch.\n"
                    f"Device: {self.device}\n"
                    f"Original error: {type(e).__name__}: {e}\n\n"
                    "If this is a Pascal/Volta GPU (1080 Ti, P100, "
                    "V100), the SDK doesn't support that hardware "
                    "out of the box. Upgrading to Ampere+ (RTX "
                    "30/40-series, A100, etc.) is the path forward; "
                    "see beak's submit-embed modal for memory and "
                    "throughput estimates per card."
                ) from e
            raise

        # Pre-Ampere GPUs (Pascal sm_60/61, Volta sm_70) have no
        # native bfloat16 support. ESMC's SDK loads weights in bf16
        # for memory efficiency on Ampere+. On Pascal those bf16
        # tensors get emulated through driver-level cast paths, and
        # certain attention kernels (specifically softmax over scaled
        # dot products) produce NaN under emulation — every embedding
        # comes out as `nan` even though the run completes "successfully".
        # Symptom: mean_embeddings.pkl loads fine, shape is correct,
        # values are uniformly NaN.
        # Fix: cast the whole model to fp32 on pre-Ampere. Roughly 2x
        # the memory but values are correct, and ESMC-300M at fp32 is
        # still ~1.2 GB activations + weights — comfortably within
        # 11 GB on a 1080 Ti.
        if self.device.type == 'cuda':
            try:
                cc = torch.cuda.get_device_capability(self.device)
            except Exception:
                cc = (0, 0)
            if cc < (8, 0):
                _log(
                    f"Pre-Ampere GPU (cc {cc[0]}.{cc[1]}) — casting "
                    f"ESMC model to float32 to avoid bf16 emulation "
                    f"NaNs in attention. Memory will be ~2x bf16 but "
                    f"still well within budget for ESMC-300M."
                )
                self.model = self.model.float()
                self._dtype = torch.float32
            else:
                self._dtype = torch.bfloat16
        else:
            # CPU path is float32 anyway; future-proof against the
            # CUDA-CPU dispatch picking a different fallback dtype.
            self.model = self.model.float()
            self._dtype = torch.float32

        self._torch = torch
        self._ESMProtein = ESMProtein
        self._LogitsConfig = LogitsConfig

    @property
    def num_layers(self) -> int:
        # The ESMC class exposes its transformer stack as
        # `self.transformer.blocks` (a ModuleList). Length = block count.
        return len(self.model.transformer.blocks)

    def normalize_layers(self, repr_layers: list) -> list:
        """Resolve negative layer indices for the ESMC indexing scheme.

        Distinct from ESM-2: the ESMC SDK returns `hidden_states` as a
        single tensor of shape `[num_layers, T, D]` — *no* leading
        initial-embedding entry — so valid indices are
        ``0..num_layers-1``. `L = -1` resolves to ``num_layers - 1``,
        i.e. the output of the final transformer block.

        This was previously sharing the ESM-2 normalisation
        (``+ L + 1``), which mapped ``-1`` to ``num_layers`` and
        crashed every sequence with
        ``IndexError: index 30 is out of bounds for dimension 0 with size 30``
        on ESMC-300M (30 blocks).
        """
        n = self.num_layers
        return [(n + L) if L < 0 else L for L in repr_layers]

    def embed_one(self, seq_id: str, seq: str, repr_layers: list) -> dict:
        protein = self._ESMProtein(sequence=seq)
        encoded = self.model.encode(protein)
        with self._torch.no_grad():
            out = self.model.logits(
                encoded,
                self._LogitsConfig(
                    sequence=True,
                    return_embeddings=True,
                    return_hidden_states=True,
                ),
            )

        # ESMC SDK API quirk: `out.hidden_states` only contains
        # special-token activations regardless of LogitsConfig flags
        # (shape `[num_layers, 1, D]`). The per-token outputs live on
        # `out.embeddings` instead — shape `[T, D]` for the last
        # layer, where T = seq_len + 2 (CLS + residues + EOS).
        #
        # That makes the last layer trivial to extract; any other
        # layer index isn't reachable through this path. Since the
        # default `repr_layers` is `[-1]` which the backend's
        # `normalize_layers` resolves to `num_layers - 1`, this covers
        # the vast majority of users. Other layer indices fall back
        # to the (broken on current SDK) hidden_states path so a
        # later SDK fix or alternate API would just work.
        last_layer = self.num_layers - 1
        seq_len = len(seq)
        result: dict = {}
        non_last = [L for L in repr_layers if L != last_layer]

        # Fast path: per-token last-layer embedding via `out.embeddings`.
        if last_layer in repr_layers:
            emb = out.embeddings  # `[T, D]` or `[1, T, D]`
            if emb.dim() == 3:
                emb = emb[0]
            T = emb.shape[0]
            if T < seq_len + 1:
                raise ValueError(
                    f"ESMC out.embeddings has T={T} but sequence "
                    f"length is {seq_len} — need T >= {seq_len + 1}."
                )
            # Slice off CLS at [0]; keep [1 : seq_len+1] for the
            # residue tokens (drop EOS at the tail).
            result[f"layer_{last_layer}"] = emb[1:seq_len + 1]

        # Slow / unreliable path: arbitrary non-last layer.
        if non_last:
            hs = out.hidden_states  # may be `[num_layers, 1, D]`
            max_idx = hs.shape[0] - 1
            bad = [L for L in non_last if L < 0 or L > max_idx]
            if bad:
                raise ValueError(
                    f"ESMC layer index(es) out of range: {bad}; "
                    f"valid range is 0..{max_idx}."
                )
            T = hs.shape[1]
            if T < seq_len + 1:
                raise ValueError(
                    f"Non-last ESMC layer extraction unsupported on "
                    f"this SDK version: hidden_states T={T} but need "
                    f">= {seq_len + 1}. Only `repr_layers=[-1]` "
                    f"(the last layer, via out.embeddings) works "
                    f"reliably; arbitrary layer indices need an SDK "
                    f"that exposes per-token hidden states."
                )
            for L in non_last:
                result[f"layer_{L}"] = hs[L, 1:seq_len + 1]

        return result


class Esm3Backend:
    """Loads ESM3 (EvolutionaryScale, 2024) via the same `esm` SDK as
    ESM-C, but with the multimodal sequence/structure/function model
    class.

    Only the open-weight variant `esm3-sm-open-v1` (1.4B params) is
    distributed for local download — it's gated behind a HuggingFace
    license-acceptance step the user does once, then the cached HF
    token in the container's `/root/.cache/huggingface/token` (mounted
    from the host via docker-compose) authenticates subsequent
    `from_pretrained` calls.

    The medium / large ESM3 variants are Forge-API only and would
    need a separate HTTP-based backend; that's not wired up here.
    """

    name = 'esm3'

    SDK_IDS = {
        'esm3_sm_open_v1': 'esm3-sm-open-v1',
    }

    def __init__(self, model_name: str, gpu_id: int):
        import torch

        self.device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        )

        # Same Pascal SDPA fallback as ESMC: pre-Ampere can't run
        # FlashAttention or memory-efficient SDPA, so route through
        # the math kernel. Without this, ESM3's attention layers can
        # silently produce NaN under emulated bf16.
        if self.device.type == 'cuda':
            try:
                cc = torch.cuda.get_device_capability(self.device)
            except Exception:
                cc = (0, 0)
            if cc < (8, 0):
                _log(
                    f"GPU compute capability {cc[0]}.{cc[1]} is "
                    f"pre-Ampere — pinning SDPA to the math kernel "
                    f"(disabling flash + mem-efficient backends)."
                )
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass

        sdk_id = self.SDK_IDS.get(model_name, model_name)
        try:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, GenerationConfig  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ESM3 requires EvolutionaryScale's `esm` SDK. The "
                "embeddings container should have it pre-installed at "
                "/opt/esmc-venv (the same venv ESM-C uses); if you're "
                "running this script from the system Python that's "
                "the bug — dispatch must use the venv interpreter for "
                "esm3_* models. "
                f"Original ImportError: {e}"
            ) from e

        try:
            self.model = ESM3.from_pretrained(sdk_id).to(self.device).eval()
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                raise RuntimeError(
                    "ESM3 weights are gated behind a HuggingFace "
                    "license-acceptance step. Visit\n"
                    "  https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1\n"
                    "while logged in, click 'Agree and access', then\n"
                    "  huggingface-cli login\n"
                    "on the host so the token is cached. The container "
                    "mounts `~/.cache/huggingface` so the cached token "
                    "is visible automatically.\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            raise

        # Pre-Ampere → fp32 cast, identical to ESMC. Without this,
        # ESM3's bf16 weights run on emulated paths on Pascal and
        # produce NaN in attention.
        if self.device.type == 'cuda':
            try:
                cc = torch.cuda.get_device_capability(self.device)
            except Exception:
                cc = (0, 0)
            if cc < (8, 0):
                _log(
                    f"Pre-Ampere GPU (cc {cc[0]}.{cc[1]}) — casting "
                    f"ESM3 model to float32 to avoid bf16 emulation "
                    f"NaNs. Memory will be ~2x bf16 (~5-7 GB for the "
                    f"1.4B-param `esm3-sm-open-v1`)."
                )
                self.model = self.model.float()
        else:
            self.model = self.model.float()

        self._torch = torch
        self._ESMProtein = ESMProtein

    @property
    def num_layers(self) -> int:
        # ESM3 exposes its transformer stack as `transformer.blocks`,
        # same name as ESM-C (the SDK shares this convention).
        return len(self.model.transformer.blocks)

    def normalize_layers(self, repr_layers: list) -> list:
        """ESM3 indexing is 0..num_layers-1 (same as ESM-C, no leading
        input-embedding row). `L = -1` → final transformer block."""
        n = self.num_layers
        return [(n + L) if L < 0 else L for L in repr_layers]

    def embed_one(self, seq_id: str, seq: str, repr_layers: list) -> dict:
        protein = self._ESMProtein(sequence=seq)
        encoded = self.model.encode(protein)
        with self._torch.no_grad():
            # `forward_and_sample` returns a `ForwardAndSampleOutput`
            # with `protein_tensor` (the multimodal token bundle) and
            # decoded outputs. For embeddings we want the per-token
            # last-layer hidden states — exposed as `output.embeddings`
            # on the lower-level forward path. Use the model's direct
            # `forward` to keep this minimal.
            out = self.model.forward(
                sequence_tokens=encoded.sequence,
                structure_tokens=getattr(encoded, "structure", None),
                ss8_tokens=getattr(encoded, "secondary_structure", None),
                sasa_tokens=getattr(encoded, "sasa", None),
                function_tokens=getattr(encoded, "function", None),
                residue_annotation_tokens=getattr(encoded, "residue_annotations", None),
            )

        # `out.embeddings` is shape `[1, T, D]` for the last layer.
        # ESM3's tokenizer prepends BOS, appends EOS — same shape
        # contract as ESM-C, so the slice convention is identical.
        emb = out.embeddings
        if emb is None:
            raise ValueError(
                "ESM3 forward returned no embeddings — model may not "
                "have been built with embedding output enabled."
            )
        if emb.dim() == 3:
            emb = emb[0]
        seq_len = len(seq)
        T = emb.shape[0]
        if T < seq_len + 1:
            raise ValueError(
                f"ESM3 embeddings has T={T} but sequence length is "
                f"{seq_len} — need T >= {seq_len + 1}."
            )

        last_layer = self.num_layers - 1
        result: dict = {}
        if last_layer in repr_layers:
            result[f"layer_{last_layer}"] = emb[1:seq_len + 1]
        non_last = [L for L in repr_layers if L != last_layer]
        if non_last:
            raise ValueError(
                f"ESM3 backend currently only supports the last layer "
                f"(layer {last_layer}); requested {non_last}. The "
                f"SDK's per-layer hidden_states API mirrors ESMC's "
                f"and could be threaded in if needed."
            )
        return result


def _make_backend(model_name: str, gpu_id: int):
    if model_name.startswith('esm2_'):
        return Esm2Backend(model_name, gpu_id)
    if model_name.startswith('esmc_') or '/esmc' in model_name.lower():
        return EsmCBackend(model_name, gpu_id)
    if model_name.startswith('esm3_') or '/esm3' in model_name.lower():
        return Esm3Backend(model_name, gpu_id)
    raise ValueError(
        f"Unrecognized model family for {model_name!r}. "
        f"Expected an esm2_*, esmc_*, or esm3_* model id."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_embeddings(
    input_fasta: str,
    output_dir: str,
    model_name: str,
    repr_layers: list,
    include_mean: bool,
    include_per_tok: bool,
    gpu_id: int,
) -> None:
    from Bio import SeqIO

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunks_mean = output_path / 'chunks' / 'mean'
    chunks_tok = output_path / 'chunks' / 'per_tok'
    progress_path = output_path / PROGRESS_FILE
    failed_path = output_path / FAILED_FILE

    _log(f"Reading sequences from {input_fasta}")
    sequences = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(input_fasta, "fasta")]
    total = len(sequences)
    _log(f"Found {total} sequences")

    # Resume: skip any seq_id whose chunk already exists for every layer we need
    def chunk_complete(seq_id: str) -> bool:
        if include_mean and not (chunks_mean / f"{seq_id}.npz").exists():
            return False
        if include_per_tok and not (chunks_tok / f"{seq_id}.npz").exists():
            return False
        return True

    pending = [(sid, seq) for sid, seq in sequences if not chunk_complete(sid)]
    already_done = total - len(pending)
    if already_done:
        _log(f"Resuming: {already_done}/{total} already embedded, {len(pending)} to go")

    progress = {
        'total': total,
        'done': already_done,
        'failed': 0,
        'current': None,
        'started_at': _now(),
        'last_update': _now(),
        'model': model_name,
    }
    _write_progress(progress_path, progress)

    if pending:
        # Fail fast when the container has no CUDA visibility. Silently
        # falling back to CPU is the worst kind of bug — an ESM-C job
        # that "looks running" on the host (a python process burning
        # cores) but is actually 100× slower than intended, with the
        # only signal being GPU utilization at 0%. Most users won't
        # notice for a long time, and large embedding sets eat days
        # of CPU before failing or being noticed.
        #
        # The most common root cause is the host's NVIDIA Container
        # Toolkit not being installed/configured: the container starts
        # fine, the GPU passthrough silently no-ops, and torch falls
        # back to CPU. Surface this as a hard error with a concrete
        # diagnostic path instead of a runtime perf cliff.
        #
        # The OS-level catch handles a different failure: torch built
        # against cu12X but the container's libcudart is from an older
        # major (the cu117 base + cu126 venv-torch combo we saw in
        # production fails here, ~1 s after submit, with no useful
        # message in the bare traceback). Re-raise with a fix-it line.
        try:
            import torch as _torch_check
        except (ImportError, OSError) as e:
            err_str = str(e).lower()
            if any(s in err_str for s in (
                'libcudart', 'libcublas', 'libcudnn', 'libnv',
                'cannot open shared object',
            )):
                _log(
                    "FATAL: torch in this venv failed to load a CUDA "
                    "runtime library — the container's base image is "
                    f"missing it.\nOriginal error: {type(e).__name__}: {e}"
                )
                _log(
                    "Cause: torch in /opt/esmc-venv was reinstalled with "
                    "a `cuXYZ` wheel that doesn't match the base image's "
                    "CUDA libs. Beak bumped the docker base image to "
                    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime "
                    "specifically to fix this; if you're seeing this, "
                    "the container was built before that change. Rebuild:"
                )
                _log(
                    "  docker compose --project-name beak_embeddings "
                    "build --no-cache embeddings"
                )
                _log(
                    "Or trigger from beak — the next embedding submission "
                    "will run `up -d --build` which picks up the new base."
                )
                raise SystemExit(2) from e
            raise
        if not _torch_check.cuda.is_available():
            msg = (
                "CUDA is not available inside the embeddings container.\n"
                "torch.cuda.is_available() == False, so this job would "
                "run on CPU and be ~100× slower than intended.\n\n"
                "Aborting before any compute is wasted. Diagnose with:\n"
                "  1) On the host: `nvidia-smi`  — confirms driver works\n"
                "  2) On the host: `docker info | grep -i nvidia`  — "
                "confirms NVIDIA Container Toolkit is installed\n"
                "  3) Inside the container: `nvidia-smi`  — confirms "
                "passthrough\n"
                "  4) Inside: `python -c 'import torch; "
                "print(torch.cuda.is_available(), torch.cuda.device_count())'`\n\n"
                "Most common fix: install the NVIDIA Container Toolkit "
                "on the host and restart docker:\n"
                "  sudo apt install -y nvidia-container-toolkit\n"
                "  sudo nvidia-ctk runtime configure --runtime=docker\n"
                "  sudo systemctl restart docker"
            )
            _log(msg)
            raise SystemExit(2)

        _log(f"Loading model: {model_name}")
        backend = _make_backend(model_name, gpu_id)
        _log(f"Backend: {backend.name}, device: {backend.device}")
        # Belt-and-suspenders: even if CUDA is "available", the chosen
        # device should resolve to cuda:N, not cpu. If we somehow
        # picked cpu (e.g., gpu_id out of range, driver hiccup), abort
        # rather than silently degrading.
        if str(backend.device) == 'cpu':
            _log(
                f"FATAL: backend resolved to device 'cpu' even though "
                f"CUDA was reported available. gpu_id={gpu_id}; "
                f"check that GPU index is valid."
            )
            raise SystemExit(2)

        # Normalize negative layer indices via the backend's own scheme.
        # ESM-2 dicts are keyed 0..num_layers (inclusive both ends), so
        # `L = -1` → num_layers. ESMC's hidden_states is a tensor of
        # shape [num_layers, T, D] (indices 0..num_layers-1), so
        # `L = -1` → num_layers - 1. Doing the math per-backend is the
        # only correct way; the earlier shared formula crashed every
        # ESMC sequence with an off-by-one IndexError.
        repr_layers = backend.normalize_layers(repr_layers)
        _log(f"Extracting representations from layers: {repr_layers}")

        for i, (seq_id, seq) in enumerate(pending, start=1):
            progress['current'] = seq_id
            progress['last_update'] = _now()
            _write_progress(progress_path, progress)

            _log(f"[{already_done + i}/{total}] {seq_id} (len={len(seq)})")

            try:
                # Preflight: both families publicly trained on <=1024 tokens.
                if len(seq) > ESM_MAX_AA:
                    raise ValueError(
                        f"sequence length {len(seq)} exceeds {ESM_MAX_AA} aa "
                        f"context window (split the sequence or use a "
                        f"long-context model)"
                    )
                # Empty sequences slip into hit FASTAs occasionally
                # (a bare `>id\n` with no body line). They contribute
                # nothing useful and trip both backends in confusing
                # ways — refuse them with a clear reason so the user
                # can see which record was malformed.
                if len(seq) == 0:
                    raise ValueError("empty sequence (no residues)")

                layer_reps = backend.embed_one(seq_id, seq, repr_layers)

                # NaN guard: an old CUDA driver vs newer PyTorch can
                # silently fall back to CPU and produce all-NaN
                # representations. Without this, the chunks pickle
                # downstream looks "successful" but every PCA / mean
                # computation explodes later with "all NaN" or
                # "inhomogeneous shape." Surface it here, on the
                # first failing sequence, with a hint pointing at the
                # log so the GPU/driver root cause is obvious.
                for k, rep in layer_reps.items():
                    rep_np = _to_numpy(rep)
                    if np.isnan(rep_np).any():
                        raise ValueError(
                            f"backend returned NaN for layer {k} — "
                            f"likely a CUDA driver / dtype mismatch "
                            f"(see warnings above; ESMC on CPU often "
                            f"produces NaN). Update the GPU driver or "
                            f"the bundled PyTorch."
                        )

                if include_mean:
                    # Reduce defensively: whatever non-embedding axes
                    # the backend returned (sequence length only, or
                    # batch+sequence, or something the SDK changed
                    # under us), collapse them all so the chunk is a
                    # 1-D `(D,)` vector. The earlier
                    # `rep.mean(0).cpu().numpy()` produced 2-D `(seq_len,
                    # D)` arrays for ESMC for reasons that traced back
                    # to an SDK shape quirk, breaking PCA assembly
                    # downstream with "inhomogeneous shape after 1
                    # dimensions." This path collapses everything
                    # safely.
                    mean_layers = {
                        k: _to_mean_vector(rep)
                        for k, rep in layer_reps.items()
                    }
                    _save_chunk(chunks_mean / f"{seq_id}.npz", mean_layers)

                if include_per_tok:
                    tok_layers = {
                        k: _to_numpy(rep)
                        for k, rep in layer_reps.items()
                    }
                    _save_chunk(chunks_tok / f"{seq_id}.npz", tok_layers)

                progress['done'] += 1

            except Exception as exc:  # noqa: BLE001 — keep the batch going
                progress['failed'] += 1
                progress['last_error'] = {
                    'seq_id': seq_id,
                    'type': type(exc).__name__,
                    'message': str(exc)[:500],
                }
                _record_failure(failed_path, seq_id, exc)
                _log(f"  ! failed: {type(exc).__name__}: {exc}")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()

            progress['last_update'] = _now()
            _write_progress(progress_path, progress)
    else:
        _log("All sequences already embedded; skipping inference.")

    progress['current'] = None
    progress['last_update'] = _now()
    _write_progress(progress_path, progress)

    # Consolidate chunks into the pickles the loader expects
    if include_mean:
        n = _consolidate_chunks(chunks_mean, output_path / MEAN_PICKLE)
        _log(f"✓ Wrote {n} mean embeddings to {output_path / MEAN_PICKLE}")

    if include_per_tok:
        n = _consolidate_chunks(chunks_tok, output_path / PER_TOK_PICKLE)
        _log(f"✓ Wrote {n} per-token embeddings to {output_path / PER_TOK_PICKLE}")

    summary = (
        f"Done: {progress['done']}/{total} embedded"
        + (f", {progress['failed']} failed (see {FAILED_FILE})" if progress['failed'] else "")
    )
    _log(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM embeddings")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="ESM model name")
    parser.add_argument("--repr-layers", nargs="+", type=int, default=[-1],
                        help="Representation layers to extract")
    parser.add_argument("--include-mean", action="store_true",
                        help="Include mean-pooled embeddings")
    parser.add_argument("--include-per-tok", action="store_true",
                        help="Include per-token embeddings")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    generate_embeddings(
        input_fasta=args.input,
        output_dir=args.output,
        model_name=args.model,
        repr_layers=args.repr_layers,
        include_mean=args.include_mean,
        include_per_tok=args.include_per_tok,
        gpu_id=args.gpu,
    )
