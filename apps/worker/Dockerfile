FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG YUME_REPO_URL=https://github.com/stdstu12/YUME
ARG YUME_COMMIT=111c3fab7fb020d1e261a68be6ec78a3fecc8d5b

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN git clone --filter=blob:none "${YUME_REPO_URL}" /opt/yume && \
    cd /opt/yume && \
    git checkout "${YUME_COMMIT}" && \
    test "$(git rev-parse HEAD)" = "${YUME_COMMIT}"

ENV PYTHONPATH=/opt/yume:${PYTHONPATH}
ENV YUME_UPSTREAM_COMMIT=${YUME_COMMIT}

WORKDIR /workspace
COPY apps/worker /workspace/apps/worker

RUN python3 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir -e /workspace/apps/worker[livekit,yume]

RUN chmod +x /workspace/apps/worker/scripts/start-worker.sh

ENTRYPOINT ["/workspace/apps/worker/scripts/start-worker.sh"]
