from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from lucid_modal import (
    build_modal_volume_commit_hook,
    create_app,
    env_secret,
    load_runtime_config_from_env,
    with_lucid_runtime,
)

from .config import WaypointRuntimeConfig
from .model import WaypointLucidModel

APP_NAME = os.getenv("MODAL_APP_NAME", "lucid-waypoint-worker")
DISPATCH_TOKEN = os.getenv("MODAL_DISPATCH_TOKEN", "")
GPU_TYPE = os.getenv("MODAL_GPU", "RTX-PRO-6000")
MODAL_MIN_CONTAINERS = int(os.getenv("MODAL_MIN_CONTAINERS", "0"))
MODAL_SCALEDOWN_WINDOW_SECS = int(os.getenv("MODAL_SCALEDOWN_WINDOW_SECS", "1200"))
MODAL_STARTUP_TIMEOUT_SECS = int(os.getenv("MODAL_STARTUP_TIMEOUT_SECS", str(60 * 60)))
CUDA_DEVEL_IMAGE = os.getenv(
    "MODAL_CUDA_DEVEL_IMAGE",
    "nvidia/cuda:12.8.0-devel-ubuntu22.04",
)
WORLD_ENGINE_REPO_URL = os.getenv(
    "WORLD_ENGINE_REPO_URL",
    "https://github.com/Overworldai/world_engine.git",
)
WORLD_ENGINE_COMMIT = os.getenv(
    "WORLD_ENGINE_COMMIT",
    "a30a00c302380c0f657347e8456bb6837ff37c22",
)

MODEL_VOLUME_NAME = os.getenv("MODAL_MODEL_VOLUME", "lucid-waypoint-models")
HF_CACHE_VOLUME_NAME = os.getenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cache_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "default"


COMPILER_CACHE_ROOT = os.getenv(
    "MODAL_COMPILER_CACHE_ROOT",
    f"/cache/huggingface/compiler/waypoint/{_cache_slug(GPU_TYPE)}",
)
image = with_lucid_runtime(
    modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential", "git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/huggingface",
            "MODAL_GPU": GPU_TYPE,
            "MODAL_HF_CACHE_VOLUME": HF_CACHE_VOLUME_NAME,
            "MODAL_COMPILER_CACHE_ROOT": COMPILER_CACHE_ROOT,
            "CUDA_CACHE_PATH": f"{COMPILER_CACHE_ROOT}/nv/ComputeCache",
            "TORCHINDUCTOR_CACHE_DIR": f"{COMPILER_CACHE_ROOT}/torchinductor",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TRITON_CACHE_DIR": f"{COMPILER_CACHE_ROOT}/triton",
        }
    )
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.12.0",
        "diffusers",
        "einops",
        "ftfy",
        "hf-xet>=1.0.0",
        "huggingface-hub==0.36.0",
        "omegaconf",
        "pillow>=10,<12",
        "rotary-embedding-torch>=0.8.8",
        "safetensors",
        "tensordict==0.10.0",
        "transformers==4.57.3",
    )
    .run_commands(
        (
            "python -m pip install --no-deps "
            f"'git+{WORLD_ENGINE_REPO_URL}@{WORLD_ENGINE_COMMIT}'"
        ),
    ),
    extra_local_dirs=[(str(PROJECT_ROOT), "/workspace/examples/waypoint_modal")],
)
image = image.run_commands("python -m pip install --no-deps /workspace/examples/waypoint_modal")
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
    "WAYPOINT_MODEL_SOURCE",
    "WAYPOINT_AE_SOURCE",
    "WAYPOINT_PROMPT_ENCODER_SOURCE",
    "WAYPOINT_DEFAULT_PROMPT",
    "WAYPOINT_SEED_IMAGE",
    "HF_HOME",
    "HF_TOKEN",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_SHOW_CPP_STACKTRACES",
    "WM_ENGINE",
    "WM_LIVEKIT_MODE",
    "WM_STATUS_TOPIC",
    "WM_FRAME_WIDTH",
    "WM_FRAME_HEIGHT",
    "WM_TARGET_FPS",
    "WM_MAX_QUEUE_FRAMES",
)


def _configure_runtime(runtime, _logger) -> None:
    if isinstance(runtime.model, WaypointLucidModel):
        runtime.model.compiler_cache_commit_hook = build_modal_volume_commit_hook()

modal_bundle = create_app(
    app_name=APP_NAME,
    model=WaypointLucidModel,
    image=image,
    gpu=GPU_TYPE,
    secrets=[runtime_secret],
    model_config_loader=WaypointRuntimeConfig.from_env,
    runtime_setup=_configure_runtime,
    min_containers=MODAL_MIN_CONTAINERS,
    scaledown_window_secs=MODAL_SCALEDOWN_WINDOW_SECS,
    startup_timeout_seconds=MODAL_STARTUP_TIMEOUT_SECS,
    volumes={"/models": model_volume, "/cache/huggingface": hf_cache_volume},
    dispatch_token=DISPATCH_TOKEN,
    runtime_config_loader=load_runtime_config_from_env,
    logger_name="waypoint_modal_example.modal",
)
app = modal_bundle.app


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    secrets=[env_secret("HF_TOKEN")],
    timeout=60 * 60,
)
def download_model(
    repo_id: str = "Overworld/Waypoint-1.1-Small",
    dest_path: str = "/models/Waypoint-1.1-Small",
    ae_repo_id: str = "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan",
    ae_dest_path: str = "/models/owl_vae_f16_c16_distill_v0_nogan",
    prompt_encoder_repo_id: str = "google/umt5-xl",
    prompt_encoder_dest_path: str = "/models/google-umt5-xl",
) -> str:
    from huggingface_hub import snapshot_download

    target = Path(dest_path)
    ae_target = Path(ae_dest_path)
    prompt_encoder_target = Path(prompt_encoder_dest_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        resume_download=True,
    )
    snapshot_download(
        repo_id=ae_repo_id,
        local_dir=str(ae_target),
        resume_download=True,
    )
    snapshot_download(
        repo_id=prompt_encoder_repo_id,
        local_dir=str(prompt_encoder_target),
        resume_download=True,
    )
    model_volume.commit()
    return str(target)
