from __future__ import annotations

import logging
from pathlib import Path

from wm_worker.yume_single_gpu_runtime import _best_output_size


def test_best_output_size_bucketizes_720p_target() -> None:
    assert _best_output_size(1280, 720, 32, 32, 1280 * 720) == (1280, 704)


def test_encode_prompt_uses_cache_for_identical_prompt() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device) -> None:
            self.moves.append(str(device))

    class FakeTextEncoder:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.calls = 0

        def __call__(self, prompts, device):
            self.calls += 1
            return [tuple(prompts), str(device), self.calls]

    class FakeTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

    from wm_worker.yume_single_gpu_runtime import YumeSingleGpuRuntime

    runtime = YumeSingleGpuRuntime(
        model_dir=Path("/tmp/yume-model"),
        frame_width=1280,
        frame_height=704,
        device="cuda",
        logger=logging.getLogger("tests.yume_runtime"),
    )
    runtime._torch = FakeTorch()
    runtime._text_encoder = FakeTextEncoder()

    first = runtime._encode_prompt("A first person walk through a city")
    second = runtime._encode_prompt("A first person walk through a city")

    assert first == second
    assert runtime._text_encoder.calls == 1
    assert runtime.prompt_cache_stats == {"hits": 1, "misses": 1, "size": 1}
