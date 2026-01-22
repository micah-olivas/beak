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

# Keep container running
CMD ["python", "-c", "print('ESM Embeddings Service Ready'); import time; time.sleep(infinity)"]