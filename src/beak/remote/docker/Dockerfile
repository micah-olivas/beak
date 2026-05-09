# Base image: CUDA 12.4 runtime + cuDNN 9 + torch 2.5.1.
#
# The previous base was pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime,
# which only ships libcudart.so.11.7. That broke the ESMC venv setup
# below: `_detect_torch_cuda_tag` on a modern host (driver CUDA 12.x)
# resolves to a `cuXYZ` ≥ cu121 wheel, but the venv-installed torch
# then crashed on import because the cu12 wheel needs libcudart.so.12
# and the container only had .so.11. Symptoms:
#   - ESMC job FAILED ~1 s after submit, before any sequence ran
#   - progress.json never advanced, failed.tsv was empty
#   - esm.log: `OSError: libcudart.so.12: cannot open shared object`
# Picking a CUDA-12 base (any 12.x — wheels forward-compat within the
# major) lines libcudart.so.12 up with whatever cu12X wheel beak's
# detector chose. cu124 covers cu121–cu126 wheels via forward-compat;
# cu128 wheels still need the strictly-less-than rule in the detector
# to step down to cu126 (already implemented).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a single pip call so the resolver can
# honour the `numpy<2` pin across ALL packages at once. Pin retained
# even though torch 2.5 supports numpy 2, because fair-esm and biotite
# still ship wheels built against numpy 1 — letting numpy 2 in causes
# the same "Numpy is not available" import error at tensor.numpy()
# time, just from a different module.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fair-esm biotite

# EvolutionaryScale's `esm` SDK (ESM-C / ESM Cambrian) lives in its
# own venv because it ships under the same top-level `esm` import as
# fair-esm — installing both into the same Python silently shadows
# whichever was installed last. Keeping fair-esm in the system Python
# (used by the ESM2 backend) and EvolutionaryScale's package in
# /opt/esmc-venv lets both backends coexist. `--system-site-packages`
# inherits numpy and the rest from the base image, but we re-install
# torch into the venv against an explicit CUDA index URL — without
# this, `pip install esm` resolves a newer-than-base torch from
# PyPI's default index, and that wheel can be built against a CUDA
# toolkit too new for the host's driver. We've seen this manifest as:
#     "The NVIDIA driver on your system is too old (found version 12080)"
# on a host that actually has driver 570.172.08 / CUDA 12.8 — the
# error means "driver too old for *this torch wheel's CUDA*", not
# "driver too old for CUDA 12.8 generally".
#
# `TORCH_CUDA_TAG` is a build arg so beak can pass the wheel that
# matches the host's driver — see `_ensure_docker_service` in
# remote/base.py for the auto-detection path. Default cu121 is a
# broadly-compatible fallback (works on any CUDA-12.x driver, 530+).
# If the requested wheel doesn't exist on PyTorch's index (e.g.
# brand-new driver, wheel not published yet), we fall back to cu121
# at build time so the image still produces a working venv.
ARG TORCH_CUDA_TAG=cu121
# Critically, the torch reinstall below does NOT pass --no-deps —
# torch's nvidia-*-cu12 transitive wheels (cuda_runtime, cublas,
# cudnn, cusparselt, …) install the actual libcudart.so.12 /
# libcublasLt.so.12 / libcusparseLt.so.0 the loader needs. With
# --no-deps the venv ends up with torch but no nvidia/<lib>/lib/
# trees, and torch's loader (which only searches the venv's own
# site-packages, not the system-site-packages it inherits via
# `--system-site-packages`) silently fails to find any of them at
# import time. Diagnosis took several iterations because `pip list`
# inside the venv lied — it sees the conda env's nvidia wheels via
# system-site-packages, but those don't reach torch's loader.
#
# nvidia-cusparselt-cu12 is also installed explicitly: torch's wheel
# metadata for some `cuXYZ` indexes doesn't pin it as a hard
# requirement, but ESMC + recent torch versions need it at import
# time anyway. Adding it to the install line is harmless when torch
# already requested it transitively.
#
# torch is pinned to >=2.5,<2.8 because:
#   - 2.5+ is the first version with stable cu126 wheels containing
#     all the `nvidia-*-cu12` deps the loader needs.
#   - <2.8 keeps Pascal (sm_61, e.g. 1080 Ti) and Volta in the
#     supported-arch list. torch 2.8+ may drop Pascal kernels — we'd
#     rather pin defensively than have a future rebuild silently
#     produce a binary that crashes on an older GPU.
#   - The PyPI fallback (`--extra-index-url`) is needed because
#     `nvidia-cusparselt-cu12` isn't always mirrored on the cuXYZ
#     PyTorch index but is on PyPI proper.
RUN python -m venv --system-site-packages /opt/esmc-venv \
    && /opt/esmc-venv/bin/pip install --no-cache-dir esm \
    && ( /opt/esmc-venv/bin/pip install --no-cache-dir \
            --force-reinstall \
            "torch>=2.5,<2.8" nvidia-cusparselt-cu12 \
            --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
            --extra-index-url https://pypi.org/simple \
       || ( echo "[beak] torch+${TORCH_CUDA_TAG} not available on PyTorch index; falling back to cu121" \
         && /opt/esmc-venv/bin/pip install --no-cache-dir \
                --force-reinstall \
                "torch>=2.5,<2.8" nvidia-cusparselt-cu12 \
                --index-url https://download.pytorch.org/whl/cu121 \
                --extra-index-url https://pypi.org/simple ) )

# Copy embedding generation script
COPY generate_embeddings.py /app/

# Keep container running — `tail -f /dev/null` blocks forever without
# the buggy `time.sleep(infinity)` (NameError crashed the container into
# a restart loop, making `docker compose exec` always find it "started"
# but never ready).
CMD ["tail", "-f", "/dev/null"]