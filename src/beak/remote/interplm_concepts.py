#!/usr/bin/env python3
"""InterPLM SAE concept annotation — runs on remote (GPU preferred).

Reproduces the asymmetric F1 evaluation from Simon & Zou 2024
(`InterPLM`, *Nat Methods* 22, 2107–2117):

  precision per amino acid, recall per domain instance.

This rewards features that fire on a single conserved residue inside a
long domain — the naive symmetric F1 buries those, which is why labels
look coarse otherwise. A feature that picks out the catalytic histidine
of every kinase scores precision=1.0 and recall=1.0 here, but recall
≈0.005 under the symmetric definition.

Pipeline
--------
1. Pull N Swiss-Prot proteins (reviewed, annotation_score=5, len 80–1024)
2. Parse each UniProtKB feature (Domain / Active site / Binding site / …)
   as a (concept_label, residue_set, domain_instance_id) tuple. Concept
   label combines feature type + description so concepts like
   ``"Binding site: ATP"`` and ``"Binding site: substrate"`` are separate.
3. Filter rare concepts (< MIN_DOMAINS instances).
4. Pass 1 over all proteins — track per-feature global max activation
   (for [0,1] normalization).
5. Pass 2 — normalize, sweep thresholds, accumulate counters:
     - tp_residues[t, f, c]  TP residue count at threshold t
     - active_per_feat[t, f] residues where feature t is active anywhere
     - domain_hits[t, f, c]  domain instances of c where feature f
                              fires in at least one residue of the span
6. F1 = 2·P·R / (P + R) where:
     P = tp_residues / active_per_feat (per-residue precision)
     R = domain_hits / total_domains_per_concept (asymmetric per-domain recall)
   Pick best threshold per (feature, concept), then best concept per feature.
7. Drop features whose best F1 < F1_MIN.

CLI
---
::

    python3 interplm_concepts.py \
        --model 8m --layer 6 --n-proteins 100 \
        --output /tmp/concepts.csv

The default `--n-proteins 10000` matches the project's full-run
configuration; pass a smaller value (100–500) for a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Match Simon & Zou's 5-threshold sweep including 0 (any positive
# activation). Threshold 0 is necessary for features that fire weakly
# but broadly — without it, those features can't pair-match concepts at
# all because their normalized acts never reach 0.15. Adding 0.6 gives
# finer granularity in the high-confidence range.
_THRESHOLDS: Tuple[float, ...] = (0.0, 0.15, 0.5, 0.6, 0.8)
_F1_MIN = 0.4
# Lowered from 10 to 5: at N=10K (vs paper's N=50K), specific concepts
# like "Active site: Proton acceptor" or rare Pfam domains have fewer
# instances. A floor of 5 keeps statistical noise reasonable while
# letting more specific concepts into the eval vocabulary.
_MIN_DOMAINS_PER_CONCEPT = 5
_PAGE_SIZE = 500   # UniProt REST per-request cap

# Annotations that aren't biologically interpretable concepts; skip them.
_SKIP_FEATURE_TYPES = {
    "Chain", "Natural variant", "Mutagenesis", "Alternative sequence",
    "Conflict", "Sequence uncertainty", "Initiator methionine",
}

_MODELS: Dict[str, Dict] = {
    "8m": {
        "esm_id":   "facebook/esm2_t6_8M_UR50D",
        "sae_repo": "Elana/InterPLM-esm2-8m",
    },
    "650m": {
        "esm_id":   "facebook/esm2_t33_650M_UR50D",
        "sae_repo": "Elana/InterPLM-esm2-650m",
    },
}


# ── Dependency bootstrap ──────────────────────────────────────────────────


def _have_gpu() -> bool:
    """Return True iff ``nvidia-smi -L`` lists at least one GPU."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and "GPU " in r.stdout
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _probe_torch() -> Optional[Tuple[str, bool]]:
    """Return (torch_version, cuda_available) or None if torch missing/broken.

    Done in a subprocess so the parent process never imports a CPU-only
    torch we'd then have to "uninstall" — Python can't unload C
    extensions, so once torch is imported in this process, switching
    builds is impossible.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.__version__); print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.strip().splitlines()
    if len(lines) < 2:
        return None
    return (lines[0].strip(), lines[1].strip() == "True")


def _ensure_deps() -> None:
    """Install torch + transformers. Use the CUDA wheel if a GPU is present.

    If a CPU-only torch is already installed but a GPU is available
    (typical on shared remote boxes that previously installed CPU torch
    for embeddings work), uninstall and replace with the CUDA wheel —
    otherwise the SAE eval falls back to CPU and runs ~50× slower.
    """
    have_gpu = _have_gpu()
    print(f"GPU available (nvidia-smi): {have_gpu}", flush=True)

    probe = _probe_torch()
    needs_install = probe is None
    if probe is not None:
        ver, cuda_ok = probe
        print(f"Existing torch: {ver}  cuda={cuda_ok}", flush=True)
        if have_gpu and not cuda_ok:
            print(
                "Replacing CPU-only torch with CUDA build "
                "(would otherwise be ~50× slower)…",
                flush=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torch"],
                check=False,
            )
            needs_install = True

    if needs_install:
        if have_gpu:
            print("Installing torch (cu121 wheel)…", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "torch",
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True,
            )
        else:
            print("Installing torch (CPU wheel)…", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "torch"],
                check=True,
            )

    try:
        __import__("transformers")
    except ImportError:
        print("Installing transformers…", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "transformers"],
            check=True,
        )


# ── Swiss-Prot fetch ──────────────────────────────────────────────────────


def _shorten_concept(label: str) -> str:
    """Truncate at first semicolon (UniProt frequently appends provenance)."""
    label = label.split(";", 1)[0].strip()
    return label[:80]


def _http_get_json(url: str, max_attempts: int = 4) -> Tuple[Dict, str]:
    """Fetch JSON with retry; returns (parsed_json, link_header).

    UniProt's paginated search responses are typically multi-MB gzip JSON
    blobs and a connection drop mid-body raises ``http.client.IncompleteRead``.
    The whole 1k/10k-protein run shouldn't die from one flaky page — retry
    with exponential backoff and reuse the same URL (UniProt cursor links
    are deterministic).
    """
    import time
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()), r.headers.get("Link", "")
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(
                f"UniProt page fetch failed "
                f"(attempt {attempt}/{max_attempts}): "
                f"{type(e).__name__}: {str(e)[:100]}",
                flush=True,
            )
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"UniProt page fetch failed after {max_attempts} attempts: {last_err}"
    )


def _fetch_proteins(n: int) -> List[Tuple[str, List[Tuple[str, List[int], int]]]]:
    """Return [(seq, [(concept, residues_0idx, domain_id), ...]), ...]."""
    print(f"Fetching {n} Swiss-Prot proteins (paginated, {_PAGE_SIZE}/page)...", flush=True)
    base_query = "reviewed:true AND annotation_score:5 AND length:[80 TO 1024]"
    next_url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode({
            "format": "json",
            "query": base_query,
            "size": _PAGE_SIZE,
        })
    )
    proteins: List[Tuple[str, List[Tuple[str, List[int], int]]]] = []
    domain_counter = 0
    while len(proteins) < n and next_url:
        data, link = _http_get_json(next_url)
        for entry in data.get("results", []):
            if len(proteins) >= n:
                break
            seq = (entry.get("sequence") or {}).get("value", "")
            if len(seq) < 50:
                continue
            domain_specs: List[Tuple[str, List[int], int]] = []
            for feat in entry.get("features", []):
                t = feat.get("type", "")
                if not t or t in _SKIP_FEATURE_TYPES:
                    continue
                loc = feat.get("location", {})
                s = (loc.get("start") or {}).get("value")
                e = (loc.get("end") or {}).get("value")
                if s is None or e is None:
                    continue
                # UniProt is 1-indexed, inclusive on both ends; convert to 0-based half-open
                residues = list(range(int(s) - 1, int(e)))
                if not residues:
                    continue
                desc = (feat.get("description") or "").strip()
                concept = _shorten_concept(f"{t}: {desc}" if desc else t)
                domain_specs.append((concept, residues, domain_counter))
                domain_counter += 1
            proteins.append((seq, domain_specs))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        next_url = m.group(1) if m else None
        print(f"  {len(proteins)}/{n} proteins, {domain_counter} domain instances",
              flush=True)
    print(f"Fetched {len(proteins)} proteins, {domain_counter} domain instances total.",
          flush=True)
    return proteins[:n]


# ── Model loading ─────────────────────────────────────────────────────────


def _download_with_retry(
    url: str, dest: str, *, label: str = "file",
    chunk: int = 4 * 1024 * 1024, max_attempts: int = 4,
) -> None:
    """Stream `url` to `dest` with chunked reads + Range-resume on partial reads.

    `urllib.request.urlopen(url).read()` fails with ``http.client.IncompleteRead``
    when the connection drops mid-body — common on 1+ GB downloads (the
    ESM-2-650M SAE is ~1.3 GB). We download in chunks and on transient
    failure resume via HTTP Range from the byte we already have.
    """
    import time

    tmp = dest + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    written = 0
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url)
            if written > 0:
                req.add_header("Range", f"bytes={written}-")
                print(
                    f"Resuming {label} from byte {written:,} "
                    f"(attempt {attempt}/{max_attempts})",
                    flush=True,
                )
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                if written > 0 and resp.status == 200:
                    # Server doesn't honour Range — start over.
                    written = 0
                mode = "ab" if written > 0 else "wb"
                with open(tmp, mode) as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        written += len(buf)
                        mb = written / (1024 * 1024)
                        if total:
                            tot_mb = (total + (written - len(buf))) / (1024 * 1024) \
                                if resp.status == 206 else total / (1024 * 1024)
                            print(
                                f"  {label}: {mb:.0f}/{tot_mb:.0f} MB",
                                flush=True,
                            )
                        else:
                            print(f"  {label}: {mb:.0f} MB", flush=True)
            os.rename(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(
                f"{label} download failed (attempt {attempt}/{max_attempts}): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            if attempt < max_attempts:
                time.sleep(2 ** attempt)  # 2, 4, 8 seconds
    raise RuntimeError(
        f"{label} download failed after {max_attempts} attempts: {last_err}"
    )


def _load_models(model_key: str, layer: int):
    """Load ESM-2 + SAE weights. Returns (esm, tokenizer, W_enc, b_enc, b_pre, device)."""
    import torch
    from transformers import AutoTokenizer, EsmModel
    cfg = _MODELS[model_key]

    print(f"Loading ESM-2 model {cfg['esm_id']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["esm_id"])
    esm = EsmModel.from_pretrained(cfg["esm_id"])
    esm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "WARNING: running on CPU. Expect ~50× slower than GPU. "
            "Either no nvidia-smi GPU is present, or the torch build "
            "doesn't have CUDA support (see _ensure_deps log above).",
            flush=True,
        )
    else:
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    esm.to(device)

    sae_path = f"/tmp/beak_sae_{model_key}_layer{layer}.pt"
    if not os.path.exists(sae_path):
        sae_url = (
            f"https://huggingface.co/{cfg['sae_repo']}/resolve/main/"
            f"layer_{layer}/ae_normalized.pt"
        )
        _download_with_retry(sae_url, sae_path, label=f"SAE layer {layer}")

    state = torch.load(sae_path, map_location=device, weights_only=True)
    W_enc = state["encoder.weight"].float()  # (n_feat, dim)
    b_enc = state["encoder.bias"].float()    # (n_feat,)
    b_pre = state["bias"].float()            # (dim,)
    print(f"SAE loaded: {W_enc.shape[0]} features, dim={W_enc.shape[1]}", flush=True)
    return esm, tokenizer, W_enc, b_enc, b_pre, device


def _embed_and_sae(
    seq: str, esm, tokenizer, W_enc, b_enc, b_pre, device, layer: int,
):
    """Compute SAE activations (L, n_feat) for one sequence (kept on device)."""
    import torch
    inp = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        out = esm(**inp, output_hidden_states=True)
    h = out.hidden_states[layer][0, 1:-1]              # (L, dim) — strip BOS/EOS
    pre = (h - b_pre.unsqueeze(0)) @ W_enc.t() + b_enc.unsqueeze(0)
    return torch.relu(pre)


# ── Two-pass concept evaluation ───────────────────────────────────────────


def _compute_concept_metrics(
    proteins, esm, tokenizer, W_enc, b_enc, b_pre, device, layer: int,
):
    """Returns list of dicts (one row per annotated feature) for the CSV."""
    import torch
    import numpy as np

    n_feat = W_enc.shape[0]
    n_thr = len(_THRESHOLDS)

    # ── Pass 1: per-feature global max activation (for [0,1] normalization) ─
    print("Pass 1/2: per-feature max activations for normalization…", flush=True)
    max_acts = torch.zeros(n_feat, device=device)
    for i, (seq, _) in enumerate(proteins):
        if i % 200 == 0:
            print(f"  {i}/{len(proteins)}", flush=True)
        acts = _embed_and_sae(seq, esm, tokenizer, W_enc, b_enc, b_pre, device, layer)
        max_acts = torch.maximum(max_acts, acts.max(dim=0).values)
    max_acts = torch.clamp(max_acts, min=1e-6)
    nonzero = int((max_acts > 1e-5).sum().item())
    print(f"  done. {nonzero}/{n_feat} features ever activated.", flush=True)

    # ── Filter rare concepts ────────────────────────────────────────────────
    domain_count_by_concept: Counter = Counter()
    for _, domains in proteins:
        for concept, _, _ in domains:
            domain_count_by_concept[concept] += 1
    concepts = sorted([
        c for c, n in domain_count_by_concept.items()
        if n >= _MIN_DOMAINS_PER_CONCEPT
    ])
    if not concepts:
        raise RuntimeError(
            f"No concepts pass min-domain filter ({_MIN_DOMAINS_PER_CONCEPT}); "
            f"raw concepts={len(domain_count_by_concept)}. "
            f"Either lower the threshold or fetch more proteins."
        )
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    n_concepts = len(concepts)
    total_domains = np.zeros(n_concepts, dtype=np.int64)
    for c, n in domain_count_by_concept.items():
        idx = concept_to_idx.get(c)
        if idx is not None:
            total_domains[idx] = n
    print(f"Concepts kept: {n_concepts} (≥{_MIN_DOMAINS_PER_CONCEPT} domains each)",
          flush=True)

    # ── Pass 2: accumulate counters at each threshold ───────────────────────
    print("Pass 2/2: accumulating per-(threshold, feature, concept) counters…",
          flush=True)
    tp_residues = torch.zeros((n_thr, n_feat, n_concepts), dtype=torch.int64,
                              device=device)
    active_per_feat = torch.zeros((n_thr, n_feat), dtype=torch.int64, device=device)
    domain_hits = torch.zeros((n_thr, n_feat, n_concepts), dtype=torch.int64,
                              device=device)
    thresholds_t = torch.tensor(_THRESHOLDS, device=device)

    for i, (seq, domains) in enumerate(proteins):
        if i % 200 == 0:
            print(f"  {i}/{len(proteins)}", flush=True)
        acts = _embed_and_sae(seq, esm, tokenizer, W_enc, b_enc, b_pre, device, layer)
        norm_acts = acts / max_acts.unsqueeze(0)         # (L, n_feat)
        L = norm_acts.shape[0]

        # per-residue concept membership
        c_res = torch.zeros((L, n_concepts), dtype=torch.bool, device=device)
        domain_specs = []
        for concept, residues, _did in domains:
            cidx = concept_to_idx.get(concept)
            if cidx is None:
                continue
            valid = [r for r in residues if 0 <= r < L]
            if not valid:
                continue
            r_t = torch.tensor(valid, device=device, dtype=torch.long)
            c_res[r_t, cidx] = True
            domain_specs.append((cidx, r_t))

        # CUDA doesn't ship integer matmul kernels (`addmm_cuda not
        # implemented for 'Long'`), so the TP-counting matmul has to run
        # in float32 and the result is cast back to int64 for accumulation.
        c_f = c_res.to(torch.float32)
        for ti in range(n_thr):
            A = norm_acts >= thresholds_t[ti]                # (L, n_feat) bool
            A_f = A.to(torch.float32)
            tp_residues[ti] += (A_f.t() @ c_f).to(torch.int64)
            active_per_feat[ti] += A.sum(dim=0).to(torch.int64)
            for cidx, r_t in domain_specs:
                hit = A[r_t, :].any(dim=0)                   # (n_feat,) bool
                domain_hits[ti, :, cidx] += hit.to(torch.int64)

    # ── F1 + best-threshold + best-concept per feature ──────────────────────
    print("Computing F1 and selecting best concept per feature…", flush=True)
    tp_residues_n = tp_residues.cpu().numpy()
    active_n = active_per_feat.cpu().numpy()
    domain_hits_n = domain_hits.cpu().numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(
            active_n[:, :, None] > 0,
            tp_residues_n / np.maximum(active_n[:, :, None], 1),
            0.0,
        )
        recall = np.where(
            total_domains[None, None, :] > 0,
            domain_hits_n / np.maximum(total_domains[None, None, :], 1),
            0.0,
        )
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )

    # best threshold per (f, c)
    best_t_idx = f1.argmax(axis=0)                                 # (n_feat, n_concepts)
    best_f1 = np.take_along_axis(f1, best_t_idx[None], axis=0)[0]  # (n_feat, n_concepts)
    best_p = np.take_along_axis(precision, best_t_idx[None], axis=0)[0]
    best_r = np.take_along_axis(recall, best_t_idx[None], axis=0)[0]

    # best concept per feature
    best_c_idx = best_f1.argmax(axis=1)                            # (n_feat,)
    rows: List[Dict] = []
    for f in range(n_feat):
        ci = int(best_c_idx[f])
        f1_val = float(best_f1[f, ci])
        if f1_val < _F1_MIN:
            continue
        rows.append({
            "feature":   f,
            "concept":   concepts[ci],
            "f1":        f"{f1_val:.4f}",
            "precision": f"{float(best_p[f, ci]):.4f}",
            "recall":    f"{float(best_r[f, ci]):.4f}",
            "n_domains": int(total_domains[ci]),
            "threshold": f"{_THRESHOLDS[int(best_t_idx[f, ci])]:.2f}",
        })
    print(f"  {len(rows)} features have at least one concept with F1 ≥ {_F1_MIN}.",
          flush=True)
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InterPLM SAE concept annotation (Simon & Zou 2024 methodology)",
    )
    parser.add_argument("--model", default="8m", choices=list(_MODELS),
                        help="ESM-2 / SAE size (default: 8m)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Transformer layer to evaluate")
    parser.add_argument("--n-proteins", type=int, default=10000,
                        help="Swiss-Prot eval set size (default: 10000; "
                             "use 100–500 for a smoke test)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    _ensure_deps()

    proteins = _fetch_proteins(args.n_proteins)
    if len(proteins) < 50:
        print(
            f"WARNING: only {len(proteins)} proteins fetched; "
            "results will be unreliable.",
            flush=True,
        )

    esm, tokenizer, W_enc, b_enc, b_pre, device = _load_models(args.model, args.layer)
    rows = _compute_concept_metrics(
        proteins, esm, tokenizer, W_enc, b_enc, b_pre, device, args.layer,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature", "concept", "f1", "precision",
                        "recall", "n_domains", "threshold"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} annotated features to {args.output}", flush=True)


if __name__ == "__main__":
    main()
