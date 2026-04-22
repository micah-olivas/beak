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


def generate_embeddings(
    input_fasta: str,
    output_dir: str,
    model_name: str,
    repr_layers: list,
    include_mean: bool,
    include_per_tok: bool,
    gpu_id: int,
) -> None:
    import esm
    import torch
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

    # Load model only if there's real work to do
    if pending:
        _log(f"Loading model: {model_name}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)

        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        _log(f"Using device: {device}")

        batch_converter = alphabet.get_batch_converter()

        for i, (seq_id, seq) in enumerate(pending, start=1):
            progress['current'] = seq_id
            progress['last_update'] = _now()
            _write_progress(progress_path, progress)

            _log(f"[{already_done + i}/{total}] {seq_id} (len={len(seq)})")

            try:
                _, batch_strs, batch_tokens = batch_converter([(seq_id, seq)])
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=repr_layers,
                                    return_contacts=False)

                seq_len = len(batch_strs[0])

                if include_mean:
                    mean_layers = {}
                    for layer in repr_layers:
                        rep = results["representations"][layer][0, 1:seq_len + 1]
                        mean_layers[f"layer_{layer}"] = rep.mean(0).cpu().numpy()
                    _save_chunk(chunks_mean / f"{seq_id}.npz", mean_layers)

                if include_per_tok:
                    tok_layers = {}
                    for layer in repr_layers:
                        rep = results["representations"][layer][0, 1:seq_len + 1]
                        tok_layers[f"layer_{layer}"] = rep.cpu().numpy()
                    _save_chunk(chunks_tok / f"{seq_id}.npz", tok_layers)

                progress['done'] += 1

            except Exception as exc:  # noqa: BLE001 — we want to continue the batch
                progress['failed'] += 1
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
