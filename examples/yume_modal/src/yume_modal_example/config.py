from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class YumeRuntimeConfig(BaseModel):
    frame_width: int = Field(default=1280, gt=0)
    frame_height: int = Field(default=720, gt=0)
    target_fps: int = Field(default=2, gt=0)
    wm_engine: str = Field(default="fake", min_length=1)
    yume_chunk_frames: int = Field(default=8, gt=0)
    yume_base_prompt: str = Field(
        default="POV of a character walking in a minecraft scene",
        min_length=1,
    )
    yume_model_dir: Path = Path("/models/Yume-5B-720P")

    @classmethod
    def from_env(cls) -> "YumeRuntimeConfig":
        return cls(
            frame_width=int(os.getenv("WM_FRAME_WIDTH", "1280")),
            frame_height=int(os.getenv("WM_FRAME_HEIGHT", "720")),
            target_fps=int(os.getenv("WM_TARGET_FPS", "2")),
            wm_engine=os.getenv("WM_ENGINE", "fake"),
            yume_chunk_frames=int(os.getenv("YUME_CHUNK_FRAMES", "8")),
            yume_base_prompt=os.getenv(
                "YUME_BASE_PROMPT",
                "POV of a character walking in a minecraft scene",
            ),
            yume_model_dir=Path(os.getenv("YUME_MODEL_DIR", "/models/Yume-5B-720P")),
        )
