FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set working directory
WORKDIR /dreamerv3_carla

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy everything including handler.py
COPY . .

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies except JAX
RUN pip install -r requirements.txt --no-cache-dir

# Install JAX (separate, with URL)
RUN pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir

# Install RunPod SDK
RUN pip install runpod --no-cache-dir

# Expose handler.py (RunPod looks for it here)
ENV PYTHONPATH=/dreamerv3_carla

CMD ["python3", "handler.py"]
