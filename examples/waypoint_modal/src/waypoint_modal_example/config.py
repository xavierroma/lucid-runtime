from __future__ import annotations

import os

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

    @classmethod
    def from_env(cls) -> "WaypointRuntimeConfig":
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
        )
