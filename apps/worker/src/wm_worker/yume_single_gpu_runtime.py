from __future__ import annotations

import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


class YumeSingleGpuRuntimeError(RuntimeError):
    pass


def _best_output_size(
    width: int,
    height: int,
    width_stride: int,
    height_stride: int,
    expected_area: int,
) -> tuple[int, int]:
    ratio = width / height
    out_width = (expected_area * ratio) ** 0.5
    out_height = expected_area / out_width

    width_first = int(out_width // width_stride * width_stride)
    height_first = int(expected_area / width_first // height_stride * height_stride)
    ratio_width_first = width_first / height_first

    height_first_alt = int(out_height // height_stride * height_stride)
    width_first_alt = int(expected_area / height_first_alt // width_stride * width_stride)
    ratio_height_first = width_first_alt / height_first_alt

    if max(ratio / ratio_width_first, ratio_width_first / ratio) < max(
        ratio / ratio_height_first, ratio_height_first / ratio
    ):
        return width_first, height_first
    return width_first_alt, height_first_alt


class YumeSingleGpuRuntime:
    """Minimal single-GPU Yume inference adapter for per-chunk sampling."""

    def __init__(
        self,
        *,
        model_dir: Path,
        frame_width: int,
        frame_height: int,
        device: str,
        logger,
    ) -> None:
        self._model_dir = model_dir
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._device = device
        self._logger = logger

        self._torch = None
        self._dtype = None
        self._sample_steps = 4
        self._sample_shift = 5.0
        self._vae_stride = (4, 16, 16)
        self._patch_size = (1, 2, 2)
        self._sample_width = frame_width
        self._sample_height = frame_height
        self._sp_size = 1

        self._wan_model = None
        self._vae = None
        self._text_encoder = None
        self._prompt_cache: OrderedDict[str, list[object]] = OrderedDict()
        self._prompt_cache_max_entries = 8
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0

    def load(self) -> None:
        total_started = time.perf_counter()
        try:
            imports_started = time.perf_counter()
            import torch
            from safetensors.torch import load_file
            from wan23.configs import WAN_CONFIGS
            from wan23.modules.model import WanModel
            from wan23.modules.t5 import T5EncoderModel
            from wan23.modules.vae2_2 import Wan2_2_VAE
        except Exception as exc:  # pragma: no cover - dependency/runtime boundary
            raise YumeSingleGpuRuntimeError(
                "failed importing Yume runtime modules"
            ) from exc
        imports_ms = (time.perf_counter() - imports_started) * 1000

        if self._device != "cuda":
            raise YumeSingleGpuRuntimeError("Yume runtime requires CUDA device")
        if self._frame_height % 16 != 0 or self._frame_width % 16 != 0:
            raise YumeSingleGpuRuntimeError(
                "frame dimensions must be divisible by 16 for Yume VAE stride"
            )

        config = WAN_CONFIGS["ti2v-5B"]
        self._torch = torch
        self._dtype = config.param_dtype
        self._sample_shift = float(getattr(config, "sample_shift", 5.0))
        self._vae_stride = tuple(int(x) for x in config.vae_stride)
        self._patch_size = tuple(int(x) for x in config.patch_size)
        self._sp_size = int(getattr(config, "sp_size", 1))
        self._sample_width, self._sample_height = _best_output_size(
            self._frame_width,
            self._frame_height,
            self._patch_size[2] * self._vae_stride[2],
            self._patch_size[1] * self._vae_stride[1],
            self._frame_width * self._frame_height,
        )

        t5_checkpoint = self._model_dir / str(config.t5_checkpoint)
        tokenizer_path = self._model_dir / str(config.t5_tokenizer)
        vae_checkpoint = self._model_dir / str(config.vae_checkpoint)
        model_weights = self._model_dir / "diffusion_pytorch_model.safetensors"
        if not t5_checkpoint.exists() or not vae_checkpoint.exists() or not model_weights.exists():
            raise YumeSingleGpuRuntimeError(
                "Yume model directory is missing required checkpoint files"
            )
        if not tokenizer_path.exists():
            raise YumeSingleGpuRuntimeError(
                f"missing tokenizer directory: {tokenizer_path}"
            )

        t5_started = time.perf_counter()
        self._text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(t5_checkpoint),
            tokenizer_path=str(tokenizer_path),
        )
        t5_ms = (time.perf_counter() - t5_started) * 1000

        vae_started = time.perf_counter()
        self._vae = Wan2_2_VAE(
            vae_pth=str(vae_checkpoint),
            device=torch.device(self._device),
        )
        vae_ms = (time.perf_counter() - vae_started) * 1000

        model_init_started = time.perf_counter()
        self._wan_model = WanModel.from_pretrained(str(self._model_dir))
        model_init_ms = (time.perf_counter() - model_init_started) * 1000

        safetensors_started = time.perf_counter()
        state_dict = load_file(str(model_weights))
        safetensors_ms = (time.perf_counter() - safetensors_started) * 1000

        load_state_started = time.perf_counter()
        self._wan_model.load_state_dict(state_dict, strict=False)
        load_state_ms = (time.perf_counter() - load_state_started) * 1000

        to_cuda_started = time.perf_counter()
        self._wan_model = (
            self._wan_model.to(device=self._device, dtype=self._dtype)
            .eval()
            .requires_grad_(False)
        )
        to_cuda_ms = (time.perf_counter() - to_cuda_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        self._logger.info(
            (
                "loaded Yume runtime from %s sample_size=%sx%s "
                "load_ms={imports:%.2f,t5:%.2f,vae:%.2f,wan_init:%.2f,"
                "safetensors:%.2f,load_state:%.2f,to_cuda:%.2f,total:%.2f}"
            ),
            self._model_dir,
            self._sample_width,
            self._sample_height,
            imports_ms,
            t5_ms,
            vae_ms,
            model_init_ms,
            safetensors_ms,
            load_state_ms,
            to_cuda_ms,
            total_ms,
        )

    def generate_chunk(self, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        if self._torch is None or self._wan_model is None or self._vae is None:
            raise YumeSingleGpuRuntimeError("runtime must be loaded before inference")
        return self._sample_text_to_video(prompt=prompt, chunk_frames=max(1, int(chunk_frames)))

    def cache_prompt(self, prompt: str) -> None:
        _ = self._encode_prompt(prompt)

    @property
    def prompt_cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._prompt_cache_hits,
            "misses": self._prompt_cache_misses,
            "size": len(self._prompt_cache),
        }

    def _sample_text_to_video(self, *, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        torch = self._torch
        assert torch is not None

        context = self._encode_prompt(prompt)
        seq_len, noise = self._build_noise(chunk_frames)
        arg_c = {"context": context, "seq_len": seq_len}

        sampling_sigmas = self._sampling_sigmas(self._sample_steps, self._sample_shift)
        latent = noise
        latent_frame_zero = noise.shape[1]
        with torch.no_grad():
            for i, sigma in enumerate(sampling_sigmas):
                timestep = torch.tensor([float(sigma * 1000.0)], device=self._device)
                latent_model_input = [latent.to(dtype=self._dtype)]
                with torch.autocast("cuda", dtype=self._dtype):
                    noise_pred = self._wan_model(
                        latent_model_input,
                        t=timestep,
                        latent_frame_zero=latent_frame_zero,
                        **arg_c,
                        flag=False,
                    )[0]

                next_sigma = 0.0 if i + 1 == len(sampling_sigmas) else float(sampling_sigmas[i + 1])
                tail = latent[:, -latent_frame_zero:, :, :]
                pred_tail = noise_pred[:, -latent_frame_zero:, :, :]
                new_tail = tail + (next_sigma - float(sigma)) * pred_tail
                prefix = latent[:, :-latent_frame_zero, :, :]
                latent = (
                    torch.cat([prefix, new_tail.to(dtype=self._dtype)], dim=1)
                    if prefix.numel() > 0
                    else new_tail.to(dtype=self._dtype)
                )

            with torch.autocast("cuda", dtype=self._dtype):
                decoded = self._vae.decode([latent])[0]

        frames = self._decode_rgb_frames(decoded, chunk_frames=chunk_frames)
        return frames

    def _build_noise(self, chunk_frames: int) -> tuple[int, object]:
        torch = self._torch
        assert torch is not None

        frame_count = chunk_frames
        target_shape = (
            self._vae.model.z_dim,
            (frame_count - 1) // self._vae_stride[0] + 1,
            self._sample_height // self._vae_stride[1],
            self._sample_width // self._vae_stride[2],
        )
        seq_len = (
            (target_shape[2] * target_shape[3])
            / (self._patch_size[1] * self._patch_size[2])
            * target_shape[1]
        )
        seq_len = int(math.ceil(seq_len / self._sp_size) * self._sp_size)
        generator = torch.Generator(device=self._device)
        generator.manual_seed(random.randint(0, 2**31 - 1))
        noise = torch.randn(
            *target_shape,
            dtype=torch.float32,
            device=self._device,
            generator=generator,
        )
        return seq_len, noise

    def _encode_prompt(self, prompt: str) -> list[object]:
        torch = self._torch
        assert torch is not None
        if self._text_encoder is None:
            raise YumeSingleGpuRuntimeError("text encoder not initialized")

        normalized_prompt = prompt.strip() or "An explorable realistic world"
        cached = self._prompt_cache.get(normalized_prompt)
        if cached is not None:
            self._prompt_cache_hits += 1
            self._prompt_cache.move_to_end(normalized_prompt)
            self._logger.info(
                "reused yume prompt embedding cache=hit prompt_len=%s cache_size=%s hits=%s misses=%s",
                len(normalized_prompt),
                len(self._prompt_cache),
                self._prompt_cache_hits,
                self._prompt_cache_misses,
            )
            return cached

        encode_started = time.perf_counter()
        try:
            self._text_encoder.model.to(self._device)
            context = self._text_encoder([normalized_prompt], torch.device(self._device))
            self._prompt_cache_misses += 1
            self._prompt_cache[normalized_prompt] = context
            self._prompt_cache.move_to_end(normalized_prompt)
            while len(self._prompt_cache) > self._prompt_cache_max_entries:
                evicted_prompt, _ = self._prompt_cache.popitem(last=False)
                self._logger.info(
                    "evicted yume prompt cache entry prompt_len=%s",
                    len(evicted_prompt),
                )
            self._logger.info(
                (
                    "encoded yume prompt cache=miss prompt_len=%s encode_ms=%.2f "
                    "cache_size=%s hits=%s misses=%s"
                ),
                len(normalized_prompt),
                (time.perf_counter() - encode_started) * 1000,
                len(self._prompt_cache),
                self._prompt_cache_hits,
                self._prompt_cache_misses,
            )
            return context
        finally:
            self._text_encoder.model.to("cpu")

    @staticmethod
    def _sampling_sigmas(steps: int, shift: float) -> np.ndarray:
        sigma = np.linspace(1, 0, steps + 1)[:steps]
        return shift * sigma / (1 + (shift - 1) * sigma)

    def _decode_rgb_frames(self, decoded_video: object, *, chunk_frames: int) -> list[np.ndarray]:
        torch = self._torch
        assert torch is not None

        pixels = decoded_video.clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8)
        # Model output is (C, F, H, W). Convert to list[(H, W, C)].
        frames = pixels.detach().cpu().numpy()
        frames = np.transpose(frames, (1, 2, 3, 0))
        if frames.size == 0:
            raise YumeSingleGpuRuntimeError("decoded empty frame tensor")

        out: list[np.ndarray] = []
        limit = min(chunk_frames, frames.shape[0])
        for idx in range(limit):
            out.append(self._fit_output_frame(frames[idx]))

        while len(out) < chunk_frames:
            out.append(out[-1].copy())

        return out

    def _fit_output_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape[0] == self._frame_height and frame.shape[1] == self._frame_width:
            return frame

        out = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)
        copy_height = min(frame.shape[0], self._frame_height)
        copy_width = min(frame.shape[1], self._frame_width)

        src_y = max((frame.shape[0] - copy_height) // 2, 0)
        src_x = max((frame.shape[1] - copy_width) // 2, 0)
        dst_y = max((self._frame_height - copy_height) // 2, 0)
        dst_x = max((self._frame_width - copy_width) // 2, 0)

        out[dst_y : dst_y + copy_height, dst_x : dst_x + copy_width] = frame[
            src_y : src_y + copy_height,
            src_x : src_x + copy_width,
        ]
        return out
