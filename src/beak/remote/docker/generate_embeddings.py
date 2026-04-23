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
    """Loads ESM-C via HuggingFace transformers + trust_remote_code.

    Picking `transformers` over EvolutionaryScale's `esm` package avoids a
    PyPI conflict with `fair-esm` (both own the top-level `esm` import),
    so ESM2 + ESM-C can coexist in the same container.
    """

    name = 'esmc'

    # Short aliases -> HF hub model IDs. Users can still pass a full
    # "org/name" path and it will be used verbatim.
    HF_IDS = {
        'esmc_300m': 'EvolutionaryScale/esmc_300m_2024_12',
        'esmc_600m': 'EvolutionaryScale/esmc_600m_2024_12',
    }

    def __init__(self, model_name: str, gpu_id: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        hf_id = self.HF_IDS.get(model_name, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        self.device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device).eval()
        self._torch = torch

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def embed_one(self, seq_id: str, seq: str, repr_layers: list) -> dict:
        inputs = self.tokenizer(
            seq, return_tensors='pt', add_special_tokens=True,
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        # transformers returns hidden_states as a tuple of length
        # num_layers + 1 (index 0 = initial embedding, index N = after
        # block N). We slice off CLS/EOS to match the ESM2 convention.
        seq_len = len(seq)
        return {
            f"layer_{L}": out.hidden_states[L][0, 1:seq_len + 1]
            for L in repr_layers
        }


def _make_backend(model_name: str, gpu_id: int):
    if model_name.startswith('esm2_'):
        return Esm2Backend(model_name, gpu_id)
    if model_name.startswith('esmc_') or '/esmc' in model_name.lower():
        return EsmCBackend(model_name, gpu_id)
    raise ValueError(
        f"Unrecognized model family for {model_name!r}. "
        f"Expected an esm2_* or esmc_* model id."
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
        _log(f"Loading model: {model_name}")
        backend = _make_backend(model_name, gpu_id)
        _log(f"Backend: {backend.name}, device: {backend.device}")

        # Normalize negative layer indices to positive. Both backends key
        # their hidden states by exact layer number; [-1] would KeyError.
        num_layers = backend.num_layers
        repr_layers = [
            (num_layers + L + 1) if L < 0 else L
            for L in repr_layers
        ]
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

                layer_reps = backend.embed_one(seq_id, seq, repr_layers)

                if include_mean:
                    mean_layers = {
                        k: rep.mean(0).cpu().numpy()
                        for k, rep in layer_reps.items()
                    }
                    _save_chunk(chunks_mean / f"{seq_id}.npz", mean_layers)

                if include_per_tok:
                    tok_layers = {
                        k: rep.cpu().numpy()
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
