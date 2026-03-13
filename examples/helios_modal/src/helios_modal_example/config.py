from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

DEFAULT_PROMPT = (
    "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, "
    "turquoise ocean."
)


def _parse_csv_ints(value: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = value.strip()
    if not raw:
        return default
    parts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not parts:
        return default
    if any(part <= 0 for part in parts):
        raise ValueError("all comma-separated integer values must be positive")
    return parts


def _parse_bool(value: str, *, default: bool) -> bool:
    raw = value.strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class HeliosRuntimeConfig(BaseModel):
    frame_width: int = Field(default=640, gt=0)
    frame_height: int = Field(default=384, gt=0)
    output_fps: int = Field(default=24, gt=0)
    wm_engine: str = Field(default="helios", min_length=1)
    helios_model_source: str = Field(default="/models/Helios-Distilled", min_length=1)
    helios_default_prompt: str = Field(default=DEFAULT_PROMPT, min_length=1)
    helios_negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT, min_length=1)
    helios_chunk_frames: int = Field(default=33, gt=0)
    helios_guidance_scale: float = Field(default=1.0, ge=0.0)
    helios_pyramid_steps: tuple[int, ...] = Field(default=(2, 2, 2))
    helios_amplify_first_chunk: bool = True
    helios_enable_group_offloading: bool = False
    helios_group_offloading_type: str = Field(default="leaf_level", min_length=1)
    helios_max_sequence_length: int = Field(default=512, gt=0)

    @field_validator("helios_group_offloading_type")
    @classmethod
    def _validate_group_offloading_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"leaf_level", "block_level"}:
            raise ValueError("helios_group_offloading_type must be leaf_level or block_level")
        return normalized

    @classmethod
    def from_env(cls) -> "HeliosRuntimeConfig":
        return cls(
            frame_width=int(os.getenv("HELIOS_FRAME_WIDTH", "640")),
            frame_height=int(os.getenv("HELIOS_FRAME_HEIGHT", "384")),
            output_fps=int(os.getenv("HELIOS_OUTPUT_FPS", "24")),
            wm_engine=os.getenv("WM_ENGINE", "helios"),
            helios_model_source=os.getenv("HELIOS_MODEL_SOURCE", "/models/Helios-Distilled"),
            helios_default_prompt=os.getenv("HELIOS_DEFAULT_PROMPT", DEFAULT_PROMPT),
            helios_negative_prompt=os.getenv("HELIOS_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT),
            helios_chunk_frames=int(os.getenv("HELIOS_CHUNK_FRAMES", "33")),
            helios_guidance_scale=float(os.getenv("HELIOS_GUIDANCE_SCALE", "1.0")),
            helios_pyramid_steps=_parse_csv_ints(
                os.getenv("HELIOS_PYRAMID_STEPS", "2,2,2"),
                default=(2, 2, 2),
            ),
            helios_amplify_first_chunk=_parse_bool(
                os.getenv("HELIOS_AMPLIFY_FIRST_CHUNK", "1"),
                default=True,
            ),
            helios_enable_group_offloading=_parse_bool(
                os.getenv("HELIOS_ENABLE_GROUP_OFFLOADING", "0"),
                default=False,
            ),
            helios_group_offloading_type=os.getenv(
                "HELIOS_GROUP_OFFLOADING_TYPE",
                "leaf_level",
            ),
            helios_max_sequence_length=int(os.getenv("HELIOS_MAX_SEQUENCE_LENGTH", "512")),
        )
