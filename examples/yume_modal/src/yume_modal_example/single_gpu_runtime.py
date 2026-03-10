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
    """Thin single-GPU adapter over upstream ``wan23.Yume`` for chunked sessions."""

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

        self._upstream_yume = None
        self._wan_model = None
        self._vae = None
        self._text_encoder = None
        self._masks_like = None
        self._prompt_cache: OrderedDict[str, list[object]] = OrderedDict()
        self._prompt_cache_max_entries = 8
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0
        self._continuation_latent = None

    def load(self) -> None:
        total_started = time.perf_counter()
        try:
            imports_started = time.perf_counter()
            import torch
            from wan23.configs import WAN_CONFIGS
            from wan23 import Yume as UpstreamYume
            from wan23.modules.model import WanAttentionBlock
            from wan23.utils.utils import masks_like
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
        self._masks_like = masks_like
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

        upstream_started = time.perf_counter()
        self._upstream_yume = UpstreamYume(
            config=config,
            checkpoint_dir=str(self._model_dir),
            device_id=0,
        )
        upstream_ms = (time.perf_counter() - upstream_started) * 1000

        self._text_encoder = self._upstream_yume.text_encoder
        self._vae = self._upstream_yume.vae
        self._wan_model = self._upstream_yume.model

        vae_started = time.perf_counter()
        try:
            for parameter in self._vae.model.parameters():
                parameter.data = parameter.data.to(self._dtype)
            self._vae.model.to(device=torch.device(self._device))
        except Exception:
            self._vae.model.to(device=torch.device(self._device))
        vae_ms = (time.perf_counter() - vae_started) * 1000

        to_cuda_started = time.perf_counter()
        self._wan_model = (
            self._wan_model.to(device=self._device, dtype=self._dtype)
            .eval()
            .requires_grad_(False)
        )
        self._wan_model.sideblock = WanAttentionBlock(
            self._wan_model.dim,
            self._wan_model.ffn_dim,
            self._wan_model.num_heads,
            self._wan_model.window_size,
            self._wan_model.qk_norm,
            self._wan_model.cross_attn_norm,
            self._wan_model.eps,
        ).to(device=self._device, dtype=self._dtype)
        self._wan_model.mask_token = torch.nn.Parameter(
            torch.zeros(
                1,
                1,
                self._wan_model.dim,
                device=self._device,
                dtype=self._dtype,
            )
        )
        torch.nn.init.normal_(self._wan_model.mask_token, std=0.02)
        to_cuda_ms = (time.perf_counter() - to_cuda_started) * 1000

        t5_to_cpu_started = time.perf_counter()
        self._text_encoder.model.to("cpu")
        t5_ms = (time.perf_counter() - t5_to_cpu_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        self._logger.info(
            (
                "loaded Yume runtime from %s via upstream wan23.Yume sample_size=%sx%s "
                "load_ms={imports:%.2f,upstream:%.2f,vae:%.2f,t5_to_cpu:%.2f,to_cuda:%.2f,total:%.2f}"
            ),
            self._model_dir,
            self._sample_width,
            self._sample_height,
            imports_ms,
            upstream_ms,
            vae_ms,
            t5_ms,
            to_cuda_ms,
            total_ms,
        )

    def generate_chunk(self, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        if self._torch is None or self._wan_model is None or self._vae is None:
            raise YumeSingleGpuRuntimeError("runtime must be loaded before inference")
        chunk_frames = max(1, int(chunk_frames))
        if self._continuation_latent is None:
            return self._sample_initial_chunk(prompt=prompt, chunk_frames=chunk_frames)
        return self._sample_continuation_chunk(prompt=prompt, chunk_frames=chunk_frames)

    def cache_prompt(self, prompt: str) -> None:
        _ = self._encode_prompt(prompt)

    def reset_session_state(self) -> None:
        self._continuation_latent = None

    @property
    def prompt_cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._prompt_cache_hits,
            "misses": self._prompt_cache_misses,
            "size": len(self._prompt_cache),
        }

    def _sample_initial_chunk(self, *, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        torch = self._torch
        assert torch is not None

        context = self._encode_prompt(prompt)
        latent_frame_zero = self._latent_frame_count(chunk_frames)
        seq_len = self._sequence_length(latent_frame_zero)
        latent = self._sample_noise(latent_frame_zero)
        arg_c = {"context": context, "seq_len": seq_len}
        with torch.no_grad():
            latent = self._denoise(
                latent=latent,
                latent_frame_zero=latent_frame_zero,
                arg_c=arg_c,
                flag=False,
            )
            with torch.autocast("cuda", dtype=self._dtype):
                decoded = self._vae.decode([latent[:, -latent_frame_zero:, :, :]])[0]

        self._continuation_latent = latent[:, -latent_frame_zero:, :, :].detach().to(
            dtype=self._dtype
        )
        return self._decode_rgb_frames(decoded, chunk_frames=chunk_frames)

    def _sample_continuation_chunk(self, *, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        torch = self._torch
        assert torch is not None
        if self._continuation_latent is None:
            raise YumeSingleGpuRuntimeError("continuation state is not initialized")
        if self._masks_like is None:
            raise YumeSingleGpuRuntimeError("Yume masks helper is not initialized")

        context = self._encode_prompt(prompt)
        latent_frame_zero = self._latent_frame_count(chunk_frames)
        history_latent = self._continuation_latent.to(dtype=torch.float32)
        latent = torch.cat(
            [history_latent, self._sample_noise(latent_frame_zero)],
            dim=1,
        )
        seq_len = self._sequence_length(latent.shape[1])
        arg_c = {"context": context, "seq_len": seq_len}
        _mask1, mask2 = self._masks_like(
            [latent],
            zero=True,
            latent_frame_zero=latent_frame_zero,
        )

        with torch.no_grad():
            latent = self._denoise(
                latent=latent,
                latent_frame_zero=latent_frame_zero,
                arg_c=arg_c,
                flag=True,
                mask2=mask2,
            )
            with torch.autocast("cuda", dtype=self._dtype):
                decoded = self._vae.decode([latent[:, -latent_frame_zero:, :, :]])[0]

        # Upstream YUME only carries the latest latent tail across continuation invocations.
        # Keeping the full accumulated latent history over-anchors generation and weakens
        # prompt changes over time.
        self._continuation_latent = latent[:, -latent_frame_zero:, :, :].detach().to(
            dtype=self._dtype
        )
        return self._decode_rgb_frames(decoded, chunk_frames=chunk_frames)

    def _denoise(
        self,
        *,
        latent,
        latent_frame_zero: int,
        arg_c: dict[str, object],
        flag: bool,
        mask2=None,
    ):
        torch = self._torch
        assert torch is not None

        sampling_sigmas = self._sampling_sigmas(self._sample_steps, self._sample_shift)
        for i, sigma in enumerate(sampling_sigmas):
            timestep: object
            if flag:
                if mask2 is None:
                    raise YumeSingleGpuRuntimeError("continuation mask is required")
                timestep = self._masked_timestep(
                    mask2=mask2,
                    latent_frame_zero=latent_frame_zero,
                    seq_len=int(arg_c["seq_len"]),
                    sigma=float(sigma),
                )
            else:
                timestep = torch.tensor([float(sigma * 1000.0)], device=self._device)
            latent_model_input = [latent.to(dtype=self._dtype)]
            with torch.autocast("cuda", dtype=self._dtype):
                noise_pred = self._wan_model(
                    latent_model_input,
                    t=timestep,
                    latent_frame_zero=latent_frame_zero,
                    **arg_c,
                    flag=flag,
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
        return latent

    def _sample_noise(self, latent_frames: int):
        torch = self._torch
        assert torch is not None

        generator = torch.Generator(device=self._device)
        generator.manual_seed(random.randint(0, 2**31 - 1))
        return torch.randn(
            *self._target_shape(latent_frames),
            dtype=torch.float32,
            device=self._device,
            generator=generator,
        )

    def _sequence_length(self, latent_frames: int) -> int:
        target_shape = self._target_shape(latent_frames)
        seq_len = (
            (target_shape[2] * target_shape[3])
            / (self._patch_size[1] * self._patch_size[2])
            * target_shape[1]
        )
        return int(math.ceil(seq_len / self._sp_size) * self._sp_size)

    def _target_shape(self, latent_frames: int) -> tuple[int, int, int, int]:
        return (
            self._vae.model.z_dim,
            latent_frames,
            self._sample_height // self._vae_stride[1],
            self._sample_width // self._vae_stride[2],
        )

    def _latent_frame_count(self, chunk_frames: int) -> int:
        return (max(1, int(chunk_frames)) - 1) // self._vae_stride[0] + 1

    def _masked_timestep(
        self,
        *,
        mask2,
        latent_frame_zero: int,
        seq_len: int,
        sigma: float,
    ):
        torch = self._torch
        assert torch is not None

        timestep = torch.tensor([sigma * 1000.0], device=self._device)
        temp_ts = mask2[0][0][:-latent_frame_zero, ::2, ::2].flatten()
        if temp_ts.numel() < seq_len:
            temp_ts = torch.cat(
                [
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.numel()) * timestep,
                ]
            )
        elif temp_ts.numel() > seq_len:
            temp_ts = temp_ts[:seq_len]
        return temp_ts.unsqueeze(0)

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
        src_h, src_w = frame.shape[:2]
        target_h = self._frame_height
        target_w = self._frame_width
        if src_h == target_h and src_w == target_w:
            return np.ascontiguousarray(frame, dtype=np.uint8)

        scale = max(target_w / max(src_w, 1), target_h / max(src_h, 1))
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = self._resize_nearest(frame, resized_w, resized_h)

        top = max((resized_h - target_h) // 2, 0)
        left = max((resized_w - target_w) // 2, 0)
        return np.ascontiguousarray(
            resized[top : top + target_h, left : left + target_w],
            dtype=np.uint8,
        )

    @staticmethod
    def _resize_nearest(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        y_idx = np.linspace(0, frame.shape[0] - 1, target_h).round().astype(np.int32)
        x_idx = np.linspace(0, frame.shape[1] - 1, target_w).round().astype(np.int32)
        return frame[y_idx][:, x_idx]
