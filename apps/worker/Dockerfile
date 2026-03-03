FROM ghcr.io/astral-sh/uv:python3.11-bookworm

WORKDIR /workspace

COPY apps/worker /workspace/apps/worker

RUN uv sync --project /workspace/apps/worker --frozen

ENTRYPOINT ["uv", "run", "--project", "/workspace/apps/worker", "wm-worker", "--worker-id", "wm-worker-1"]
