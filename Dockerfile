FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set working directory
WORKDIR /dreamerv3_carla

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy everything including handler.py
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

# Expose handler.py (RunPod looks for it here)
ENV PYTHONPATH=/dreamerv3_carla

CMD ["python3", "handler.py"]
