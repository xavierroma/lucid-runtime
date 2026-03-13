from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from lucid_modal import create_app, env_secret, load_runtime_config_from_env, with_lucid_runtime

from .config import HeliosRuntimeConfig
from .model import HeliosLucidModel

APP_NAME = os.getenv("MODAL_APP_NAME", "lucid-helios-worker")
DISPATCH_TOKEN = os.getenv("MODAL_DISPATCH_TOKEN", "")
GPU_TYPE = os.getenv("MODAL_GPU", "H100")
MODAL_MIN_CONTAINERS = int(os.getenv("MODAL_MIN_CONTAINERS", "0"))
MODAL_SCALEDOWN_WINDOW_SECS = int(os.getenv("MODAL_SCALEDOWN_WINDOW_SECS", "1200"))
MODAL_STARTUP_TIMEOUT_SECS = int(os.getenv("MODAL_STARTUP_TIMEOUT_SECS", str(60 * 60)))
CUDA_DEVEL_IMAGE = os.getenv(
    "MODAL_CUDA_DEVEL_IMAGE",
    "nvidia/cuda:12.8.0-devel-ubuntu22.04",
)
DIFFUSERS_REPO_URL = os.getenv(
    "DIFFUSERS_REPO_URL",
    "https://github.com/huggingface/diffusers.git",
)

MODEL_VOLUME_NAME = os.getenv("MODAL_MODEL_VOLUME", "lucid-helios-models")
HF_CACHE_VOLUME_NAME = os.getenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

image = with_lucid_runtime(
    modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential", "git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/huggingface",
            "MODAL_GPU": GPU_TYPE,
            "MODAL_HF_CACHE_VOLUME": HF_CACHE_VOLUME_NAME,
        }
    )
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate>=1.12.0",
        "ftfy",
        "huggingface-hub[hf_transfer]>=0.36.0",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
        "transformers>=4.57.0",
    )
    .run_commands(f"python -m pip install --no-deps 'git+{DIFFUSERS_REPO_URL}'"),
    extra_local_dirs=[(str(PROJECT_ROOT), "/workspace/examples/helios_modal")],
)
image = image.run_commands("python -m pip install --no-deps /workspace/examples/helios_modal")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
download_image = with_lucid_runtime(
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("huggingface-hub>=0.36.0"),
    include_livekit=False,
)

runtime_secret = env_secret(
    "MODAL_DISPATCH_TOKEN",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "HELIOS_MODEL_SOURCE",
    "HELIOS_DEFAULT_PROMPT",
    "HELIOS_NEGATIVE_PROMPT",
    "HELIOS_FRAME_WIDTH",
    "HELIOS_FRAME_HEIGHT",
    "HELIOS_OUTPUT_FPS",
    "HELIOS_CHUNK_FRAMES",
    "HELIOS_GUIDANCE_SCALE",
    "HELIOS_PYRAMID_STEPS",
    "HELIOS_AMPLIFY_FIRST_CHUNK",
    "HELIOS_ENABLE_GROUP_OFFLOADING",
    "HELIOS_GROUP_OFFLOADING_TYPE",
    "HELIOS_MAX_SEQUENCE_LENGTH",
    "HF_HOME",
    "HF_TOKEN",
    "WM_ENGINE",
    "WM_LIVEKIT_MODE",
    "WM_STATUS_TOPIC",
    "WM_MAX_QUEUE_FRAMES",
)

modal_bundle = create_app(
    app_name=APP_NAME,
    model=HeliosLucidModel,
    image=image,
    gpu=GPU_TYPE,
    secrets=[runtime_secret],
    model_config_loader=HeliosRuntimeConfig.from_env,
    min_containers=MODAL_MIN_CONTAINERS,
    scaledown_window_secs=MODAL_SCALEDOWN_WINDOW_SECS,
    startup_timeout_seconds=MODAL_STARTUP_TIMEOUT_SECS,
    volumes={"/models": model_volume, "/cache/huggingface": hf_cache_volume},
    dispatch_token=DISPATCH_TOKEN,
    runtime_config_loader=load_runtime_config_from_env,
    logger_name="helios_modal_example.modal",
)
app = modal_bundle.app


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    secrets=[env_secret("HF_TOKEN")],
    timeout=60 * 60,
)
def download_model(
    repo_id: str = "BestWishYsh/Helios-Distilled",
    dest_path: str = "/models/Helios-Distilled",
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
