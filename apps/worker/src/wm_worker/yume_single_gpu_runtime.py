from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np


class YumeSingleGpuRuntimeError(RuntimeError):
    pass


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

        self._wan_model = None
        self._vae = None
        self._text_encoder = None

    def load(self) -> None:
        try:
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

        self._text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(t5_checkpoint),
            tokenizer_path=str(tokenizer_path),
        )
        self._vae = Wan2_2_VAE(vae_pth=str(vae_checkpoint), device=torch.device(self._device))
        self._wan_model = WanModel.from_pretrained(str(self._model_dir))
        state_dict = load_file(str(model_weights))
        self._wan_model.load_state_dict(state_dict, strict=False)
        self._wan_model = (
            self._wan_model.to(device=self._device, dtype=self._dtype)
            .eval()
            .requires_grad_(False)
        )
        self._logger.info("loaded Yume runtime from %s", self._model_dir)

    def generate_chunk(self, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        if self._torch is None or self._wan_model is None or self._vae is None:
            raise YumeSingleGpuRuntimeError("runtime must be loaded before inference")
        return self._sample_text_to_video(prompt=prompt, chunk_frames=max(1, int(chunk_frames)))

    def _sample_text_to_video(self, *, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        torch = self._torch
        assert torch is not None

        context = self._encode_prompt(prompt)
        seq_len, noise = self._build_noise(chunk_frames)
        arg_c = {"context": context, "seq_len": seq_len}

        sampling_sigmas = self._sampling_sigmas(self._sample_steps, self._sample_shift)
        latent = noise
        with torch.no_grad():
            for i, sigma in enumerate(sampling_sigmas):
                timestep = torch.tensor([float(sigma * 1000.0)], device=self._device)
                latent_model_input = [latent.to(dtype=self._dtype)]
                with torch.autocast("cuda", dtype=self._dtype):
                    noise_pred = self._wan_model(
                        latent_model_input,
                        t=timestep,
                        latent_frame_zero=latent.shape[1],
                        **arg_c,
                        flag=False,
                    )[0]

                next_sigma = 0.0 if i + 1 == len(sampling_sigmas) else float(sampling_sigmas[i + 1])
                latent = latent + (next_sigma - float(sigma)) * noise_pred

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
            self._frame_height // self._vae_stride[1],
            self._frame_width // self._vae_stride[2],
        )
        seq_len = math.ceil(
            (
                (target_shape[2] * target_shape[3])
                / (self._patch_size[1] * self._patch_size[2])
                * target_shape[1]
            )
        )
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
        try:
            self._text_encoder.model.to(self._device)
            context = self._text_encoder([normalized_prompt], torch.device(self._device))
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
            out.append(np.ascontiguousarray(frames[idx], dtype=np.uint8))

        while len(out) < chunk_frames:
            out.append(out[-1].copy())

        return out
