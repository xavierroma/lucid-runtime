from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

YUME_FRAME_WIDTH = 1280
YUME_FRAME_HEIGHT = 720
YUME_OUTPUT_FPS = 2


class YumeRuntimeConfig(BaseModel):
    backend: Literal["real", "fake"] = "real"
    yume_chunk_frames: int = Field(default=8, gt=0)
    yume_base_prompt: str = Field(
        default="POV of a character walking in a minecraft scene",
        min_length=1,
    )
    yume_model_dir: Path = Path("/models/Yume-5B-720P")

    @classmethod
    def from_env(cls) -> "YumeRuntimeConfig":
        return cls(
            yume_chunk_frames=int(os.getenv("YUME_CHUNK_FRAMES", "8")),
            yume_base_prompt=os.getenv(
                "YUME_BASE_PROMPT",
                "POV of a character walking in a minecraft scene",
            ),
            yume_model_dir=Path(os.getenv("YUME_MODEL_DIR", "/models/Yume-5B-720P")),
        )
