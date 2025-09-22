# Use NVIDIA CUDA base image (with PyTorch support)
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive issues
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /dreamerv3_carla

# Copy your repository into the container
COPY . /dreamerv3_carla

# Install Python dependencies
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Default command (can be overridden in RunPod UI)
CMD ["python3", "train_carla.py"]
