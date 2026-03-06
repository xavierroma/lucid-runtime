from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import modal

from wm_worker.config import RuntimeConfig, SessionConfig
from wm_worker.modal_dispatch_api import LaunchRequest, create_app
from wm_worker.models import Assignment
from wm_worker.session_runner import SessionRunner
from wm_worker.yume_engine import YumeEngine

APP_NAME = os.getenv("MODAL_APP_NAME", "lucid-runtime-worker")
DISPATCH_TOKEN = os.getenv("MODAL_DISPATCH_TOKEN", "")
GPU_TYPE = os.getenv("MODAL_GPU", "A100")
MODAL_MIN_CONTAINERS = int(os.getenv("MODAL_MIN_CONTAINERS", "1"))
MODAL_SCALEDOWN_WINDOW_SECS = int(os.getenv("MODAL_SCALEDOWN_WINDOW_SECS", "1200"))
CUDA_DEVEL_IMAGE = os.getenv(
    "MODAL_CUDA_DEVEL_IMAGE", "nvidia/cuda:12.1.1-devel-ubuntu22.04"
)
YUME_REPO_URL = os.getenv("YUME_REPO_URL", "https://github.com/stdstu12/YUME")
YUME_COMMIT = os.getenv(
    "YUME_COMMIT", "111c3fab7fb020d1e261a68be6ec78a3fecc8d5b"
)

MODEL_VOLUME_NAME = os.getenv("MODAL_MODEL_VOLUME", "lucid-yume-models")
HF_CACHE_VOLUME_NAME = os.getenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python="3.10")
    .apt_install("build-essential", "git", "ffmpeg", "ca-certificates")
    .run_commands(
        f"git clone --filter=blob:none {YUME_REPO_URL} /opt/yume",
        f"cd /opt/yume && git checkout {YUME_COMMIT}",
        f'test "$(cd /opt/yume && git rev-parse HEAD)" = "{YUME_COMMIT}"',
    )
    .env(
        {
            "PYTHONPATH": "/opt/yume",
            "YUME_UPSTREAM_COMMIT": YUME_COMMIT,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "LOCAL_RANK": "0",
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "MAX_JOBS": "4",
        }
    )
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("packaging", "ninja", "wheel")
    .run_commands("pip install flash-attn==2.7.0.post2 --no-build-isolation")
    .pip_install(
        "accelerate==1.0.1",
        "diffusers>=0.32,<0.33",
        "transformers>=4.46,<4.47",
        "einops>=0.8,<0.9",
        "decord>=0.6,<0.7",
        "safetensors>=0.5,<0.6",
        "easydict>=1.13,<2",
        "ftfy>=6.3,<7",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "peft==0.13.2",
        "sentencepiece>=0.2,<0.3",
    )
    .pip_install_from_pyproject(
        "apps/worker/pyproject.toml",
        optional_dependencies=["livekit", "modal"],
    )
    .add_local_python_source("wm_worker", copy=True)
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "fastapi>=0.115,<1",
        "pydantic>=2.8,<3",
        "numpy>=1.26,<3",
        "httpx>=0.27,<1",
    )
    .add_local_python_source("wm_worker")
)


def _env_secret(*names: str) -> modal.Secret:
    payload = {}
    for name in names:
        value = os.getenv(name)
        if value:
            payload[name] = value
    return modal.Secret.from_dict(payload)


runtime_secret = _env_secret(
    "MODAL_DISPATCH_TOKEN",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "YUME_MODEL_DIR",
    "HF_HOME",
    "HF_TOKEN",
    "WM_ENGINE",
    "WM_LIVEKIT_MODE",
    "WM_STATUS_TOPIC",
    "WM_FRAME_WIDTH",
    "WM_FRAME_HEIGHT",
    "WM_TARGET_FPS",
    "YUME_CHUNK_FRAMES",
    "YUME_MAX_QUEUE_FRAMES",
    "YUME_BASE_PROMPT",
)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("wm_worker.modal")
    logger.setLevel(logging.INFO)
    return logger


def _load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig.from_env()


def _encode_jwt(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header_segment = _b64url(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_segment = _b64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_segment = _b64url(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def _mint_worker_access_token(
    *,
    room_name: str,
    session_id: str,
    worker_id: str,
) -> str | None:
    api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None

    now = int(time.time())
    return _encode_jwt(
        {
            "iss": api_key,
            "sub": f"{worker_id}-{session_id}",
            "nbf": now,
            "exp": now + 60 * 60,
            "video": {
                "roomJoin": True,
                "room": room_name,
            },
        },
        api_secret,
    )


def _build_assignment(request: LaunchRequest) -> Assignment:
    worker_id = request.worker_id
    return Assignment(
        session_id=request.session_id,
        room_name=request.room_name,
        worker_access_token=_mint_worker_access_token(
            room_name=request.room_name,
            session_id=request.session_id,
            worker_id=worker_id,
        )
        or request.worker_access_token,
        video_track_name=request.video_track_name,
        control_topic=request.control_topic,
    )


def _normalize_function_call_status(raw_status: Any) -> FunctionCallStatus:
    normalized = getattr(raw_status, "name", None) or str(raw_status)
    normalized = normalized.split(".")[-1].upper()
    try:
        return FunctionCallStatus(normalized)
    except ValueError:
        return FunctionCallStatus.PENDING


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 2,
    startup_timeout=20 * 60,
    min_containers=MODAL_MIN_CONTAINERS,
    max_containers=1,
    scaledown_window=MODAL_SCALEDOWN_WINDOW_SECS,
    volumes={"/models": model_volume, "/cache/huggingface": hf_cache_volume},
    secrets=[runtime_secret],
)
class WarmSessionWorker:
    @modal.enter()
    async def load(self) -> None:
        started = time.perf_counter()
        self._logger = _build_logger()
        self._runtime_config = _load_runtime_config()
        self._engine = YumeEngine(self._runtime_config, self._logger)
        await self._engine.load()
        self._logger.info(
            (
                "warm session worker ready wm_engine=%s "
                "min_containers=%s scaledown_window_secs=%s init_ms=%.2f"
            ),
            self._runtime_config.wm_engine,
            MODAL_MIN_CONTAINERS,
            MODAL_SCALEDOWN_WINDOW_SECS,
            (time.perf_counter() - started) * 1000,
        )

    @modal.method()
    async def run_session(self, payload: dict[str, Any]) -> None:
        request = LaunchRequest.model_validate(payload)
        session_config = SessionConfig.from_values(
            worker_id=request.worker_id,
            coordinator_base_url=request.coordinator_base_url,
            worker_internal_token=request.coordinator_internal_token,
        )
        runner = SessionRunner(
            self._runtime_config,
            session_config,
            self._logger,
            engine=self._engine,
        )
        try:
            await runner.run_session(_build_assignment(request))
        finally:
            await runner.close()


def _spawn_session(payload: LaunchRequest) -> str:
    function_call = WarmSessionWorker().run_session.spawn(payload.model_dump())
    function_call_id = getattr(function_call, "object_id", "")
    return function_call_id or str(function_call)


class ModalSessionDispatcher:
    def launch(self, payload: LaunchRequest) -> str:
        return _spawn_session(payload)

    def cancel(self, function_call_id: str, *, force: bool = False) -> None:
        modal.FunctionCall.from_id(function_call_id).cancel(terminate_containers=force)

    def status(self, function_call_id: str) -> FunctionCallStatus:
        try:
            graph = modal.FunctionCall.from_id(function_call_id).get_call_graph()
        except Exception as exc:
            if exc.__class__.__name__ == "NotFoundError":
                return FunctionCallStatus.NOT_FOUND
            raise

        if not graph:
            return FunctionCallStatus.PENDING

        root = next(
            (
                node
                for node in graph
                if getattr(node, "parent_input_id", None) in (None, "")
            ),
            graph[0],
        )
        return _normalize_function_call_status(getattr(root, "status", "PENDING"))


@app.function(image=image, gpu=GPU_TYPE, timeout=10 * 60, secrets=[runtime_secret])
def flash_attn_smoke() -> None:
    import flash_attn
    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    q = torch.randn(1, 4, 2, 16, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 2, 16, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 4, 2, 16, dtype=torch.float16, device="cuda")
    out = flash_attn_func(q, k, v)
    print(
        json.dumps(
            {
                "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
                "torch_version": torch.__version__,
                "cuda": torch.version.cuda or "unknown",
                "out_shape": str(tuple(out.shape)),
            }
        )
    )


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    secrets=[_env_secret("HF_TOKEN")],
    timeout=60 * 60,
)
def download_model(
    repo_id: str = "stdstu123/Yume-5B-720P",
    dest_path: str = "/models/Yume-5B-720P",
) -> str:
    from huggingface_hub import snapshot_download

    target = Path(dest_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        resume_download=True,
    )
    model_volume.commit()
    return str(target)


@app.function(image=image, secrets=[runtime_secret])
@modal.asgi_app()
def dispatch_api():
    return create_app(
        ModalSessionDispatcher(),
        os.getenv("MODAL_DISPATCH_TOKEN", DISPATCH_TOKEN),
    )
