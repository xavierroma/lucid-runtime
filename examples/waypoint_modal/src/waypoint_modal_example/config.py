from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

WAYPOINT_FRAME_WIDTH = 640
WAYPOINT_FRAME_HEIGHT = 360
WAYPOINT_OUTPUT_FPS = 20


class WaypointRuntimeConfig(BaseModel):
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
