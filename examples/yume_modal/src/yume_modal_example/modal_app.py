from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

from lucid_modal import create_app, env_secret, load_runtime_config_from_env, with_lucid_runtime

from .config import YumeRuntimeConfig
from .model import YumeLucidModel

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
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
image = with_lucid_runtime(
    modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential", "git")
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
    ),
    extra_local_dirs=[("examples/yume_modal", "/workspace/examples/yume_modal")],
)
image = image.run_commands("python -m pip install --no-deps /workspace/examples/yume_modal")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
download_image = with_lucid_runtime(
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("huggingface_hub[hf_transfer]"),
    include_livekit=False,
)

runtime_secret = env_secret(
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
    "WM_MAX_QUEUE_FRAMES",
    "YUME_CHUNK_FRAMES",
    "YUME_BASE_PROMPT",
)

modal_bundle = create_app(
    app_name=APP_NAME,
    model=YumeLucidModel,
    image=image,
    gpu=GPU_TYPE,
    secrets=[runtime_secret],
    model_config_loader=YumeRuntimeConfig.from_env,
    min_containers=MODAL_MIN_CONTAINERS,
    scaledown_window_secs=MODAL_SCALEDOWN_WINDOW_SECS,
    volumes={"/models": model_volume, "/cache/huggingface": hf_cache_volume},
    dispatch_token=DISPATCH_TOKEN,
    runtime_config_loader=load_runtime_config_from_env,
    logger_name="yume_modal_example.modal",
)
app = modal_bundle.app


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
    secrets=[env_secret("HF_TOKEN")],
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
