from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class WaypointRuntimeConfig(BaseModel):
    frame_width: int = Field(default=640, gt=0)
    frame_height: int = Field(default=360, gt=0)
    target_fps: int = Field(default=20, gt=0)
    wm_engine: str = Field(default="waypoint", min_length=1)
    waypoint_model_source: str = Field(default="/models/Waypoint-1.1-Small", min_length=1)
    waypoint_ae_source: str = Field(
        default="/models/owl_vae_f16_c16_distill_v0_nogan",
        min_length=1,
    )
    waypoint_prompt_encoder_source: str = Field(default="/models/google-umt5-xl", min_length=1)
    waypoint_default_prompt: str = Field(
        default="An explorable world with coherent geometry, stable lighting, and smooth forward motion.",
        min_length=1,
    )
    waypoint_seed_image: Path | None = None

    @classmethod
    def from_env(cls) -> "WaypointRuntimeConfig":
        raw_seed_image = os.getenv("WAYPOINT_SEED_IMAGE", "").strip()
        return cls(
            frame_width=int(os.getenv("WM_FRAME_WIDTH", "640")),
            frame_height=int(os.getenv("WM_FRAME_HEIGHT", "360")),
            target_fps=int(os.getenv("WM_TARGET_FPS", "20")),
            wm_engine=os.getenv("WM_ENGINE", "waypoint"),
            waypoint_model_source=os.getenv("WAYPOINT_MODEL_SOURCE", "/models/Waypoint-1.1-Small"),
            waypoint_ae_source=os.getenv(
                "WAYPOINT_AE_SOURCE",
                "/models/owl_vae_f16_c16_distill_v0_nogan",
            ),
            waypoint_prompt_encoder_source=os.getenv(
                "WAYPOINT_PROMPT_ENCODER_SOURCE",
                "/models/google-umt5-xl",
            ),
            waypoint_default_prompt=os.getenv(
                "WAYPOINT_DEFAULT_PROMPT",
                "An explorable world with coherent geometry, stable lighting, and smooth forward motion.",
            ),
            waypoint_seed_image=Path(raw_seed_image) if raw_seed_image else None,
        )
