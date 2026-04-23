FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a single pip call so the resolver can
# honour the `numpy<2` pin across ALL packages at once. Previously the
# numpy pin from requirements.txt was silently upgraded by the later
# `pip install fair-esm biotite` step (transitive numpy>=2 dep),
# producing "RuntimeError: Numpy is not available" when torch 2.0.1
# (compiled against numpy 1.x) tried to call .numpy() on a tensor.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fair-esm biotite

# Copy embedding generation script
COPY generate_embeddings.py /app/

# Keep container running — `tail -f /dev/null` blocks forever without
# the buggy `time.sleep(infinity)` (NameError crashed the container into
# a restart loop, making `docker compose exec` always find it "started"
# but never ready).
CMD ["tail", "-f", "/dev/null"]