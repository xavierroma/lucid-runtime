from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from yume_modal_example.single_gpu_runtime import _best_output_size


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

    from yume_modal_example.single_gpu_runtime import YumeSingleGpuRuntime

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


def test_continuation_keeps_only_latest_latent_tail() -> None:
    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.array(values, dtype=np.float32)

        @property
        def shape(self):
            return self.values.shape

        def to(self, dtype=None, device=None):
            _ = device
            if dtype is None:
                return FakeTensor(self.values.copy())
            return FakeTensor(self.values.astype(np.float32))

        def detach(self):
            return FakeTensor(self.values.copy())

        def __getitem__(self, item):
            return FakeTensor(self.values[item])

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def cat(tensors, dim):
            return FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

        @staticmethod
        def no_grad():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    _ = exc_type
                    _ = exc
                    _ = tb
                    return False

            return _Context()

        @staticmethod
        def autocast(_device_type, dtype=None):
            _ = dtype

            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    _ = exc_type
                    _ = exc
                    _ = tb
                    return False

            return _Context()

    class FakeVae:
        class model:
            z_dim = 1

        @staticmethod
        def decode(values):
            return [values[0]]

    from yume_modal_example.single_gpu_runtime import YumeSingleGpuRuntime

    runtime = YumeSingleGpuRuntime(
        model_dir=Path("/tmp/yume-model"),
        frame_width=1280,
        frame_height=704,
        device="cuda",
        logger=logging.getLogger("tests.yume_runtime"),
    )
    runtime._torch = FakeTorch()
    runtime._dtype = "float16"
    runtime._vae = FakeVae()
    runtime._text_encoder = object()
    runtime._masks_like = lambda tensors, zero, latent_frame_zero: (None, "mask")
    runtime._encode_prompt = lambda prompt: [prompt]  # type: ignore[method-assign]
    runtime._sample_noise = lambda latent_frames: FakeTensor(  # type: ignore[method-assign]
        np.full((1, latent_frames, 1, 1), 9.0, dtype=np.float32)
    )
    runtime._denoise = lambda **kwargs: kwargs["latent"]  # type: ignore[method-assign]
    runtime._decode_rgb_frames = lambda decoded, chunk_frames: [decoded] * chunk_frames  # type: ignore[method-assign]
    runtime._continuation_latent = FakeTensor([[[[1.0]], [[2.0]], [[3.0]]]])

    runtime._sample_continuation_chunk(prompt="new prompt", chunk_frames=5)

    assert runtime._continuation_latent.shape == (1, 2, 1, 1)
    assert runtime._continuation_latent.values[:, :, 0, 0].tolist() == [[9.0, 9.0]]
