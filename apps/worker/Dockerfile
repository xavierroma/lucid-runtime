FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY apps/worker /workspace/apps/worker

RUN python3 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir -e /workspace/apps/worker[livekit,yume]

RUN chmod +x /workspace/apps/worker/scripts/start-worker.sh

ENTRYPOINT ["/workspace/apps/worker/scripts/start-worker.sh"]
