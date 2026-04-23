FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ESM
RUN pip install fair-esm biotite

# Copy embedding generation script
COPY generate_embeddings.py /app/

# Keep container running — `tail -f /dev/null` blocks forever without
# the buggy `time.sleep(infinity)` (NameError crashed the container into
# a restart loop, making `docker compose exec` always find it "started"
# but never ready).
CMD ["tail", "-f", "/dev/null"]