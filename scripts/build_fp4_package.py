#!/usr/bin/env python3
"""
Build a standalone LingBot-World ACT FP4 package.

This script produces a self-contained output directory containing:
- runtime scaffold (`wan/`, `generate_prequant.py`, `load_prequant.py`)
- tokenizer + T5 + VAE assets
- pre-quantized FP4 diffusion checkpoints for high/low noise models
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import HfApi, snapshot_download


RUNTIME_ALLOW_PATTERNS = [
    "wan/**",
    "generate_prequant.py",
    "requirements.txt",
]

REQUIRED_SOURCE_FILES = [
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/tokenizer.json",
    "high_noise_model/config.json",
    "low_noise_model/config.json",
]

REQUIRED_PY_MODULES = [
    "torch",
    "bitsandbytes",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "einops",
    "imageio",
    "PIL",
    "numpy",
    "tqdm",
]


class BuildError(RuntimeError):
    """Explicit build failure exception."""



def parse_bool(value: str) -> bool:
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if value_normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LingBot-World ACT FP4 package")
    parser.add_argument(
        "--source-repo",
        default="robbyant/lingbot-world-base-act",
        help="Source Hugging Face model repo",
    )
    parser.add_argument(
        "--runtime-scaffold-repo",
        default="cahlen/lingbot-world-base-cam-nf4",
        help="Reference runtime scaffold repo with wan runtime files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for standalone FP4 package",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for quantization",
    )
    parser.add_argument(
        "--compute-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16"],
        help="Compute dtype for 4-bit layers",
    )
    parser.add_argument(
        "--double-quant",
        type=parse_bool,
        default=True,
        help="Enable nested quantization of quantization statistics (true/false)",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional Hugging Face cache root (overrides HF_HOME)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional explicit snapshot cache directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    return parser


def preflight_environment(args: argparse.Namespace) -> Tuple[object, object, object, str]:
    missing = [m for m in REQUIRED_PY_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise BuildError(
            "Missing required Python modules: "
            + ", ".join(sorted(missing))
            + ". Install dependencies on your AWS CUDA host first."
        )

    torch = importlib.import_module("torch")
    bnb = importlib.import_module("bitsandbytes")
    _diffusers = importlib.import_module("diffusers")

    if not args.device.startswith("cuda"):
        raise BuildError("--device must be a CUDA device (e.g. cuda:0)")
    if not torch.cuda.is_available():
        raise BuildError("CUDA is not available. Run this on a CUDA-enabled AWS instance.")

    device = torch.device(args.device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise BuildError(
            f"Requested device {args.device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )

    resolved_compute_dtype_name = resolve_compute_dtype_name(torch, args.compute_dtype)

    hf_home = (
        Path(args.hf_home).expanduser().resolve()
        if args.hf_home
        else Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser().resolve()
    )
    hf_home.mkdir(parents=True, exist_ok=True)
    if not os.access(hf_home, os.W_OK):
        raise BuildError(f"HF_HOME is not writable: {hf_home}")
    os.environ["HF_HOME"] = str(hf_home)

    print("[preflight] environment OK")
    print(f"[preflight] torch={torch.__version__} bitsandbytes={bnb.__version__}")
    print(f"[preflight] device={args.device} compute_dtype={resolved_compute_dtype_name}")
    print(f"[preflight] HF_HOME={hf_home}")

    return torch, bnb, _diffusers, resolved_compute_dtype_name



def resolve_compute_dtype_name(torch_mod: object, requested: str) -> str:
    if requested == "auto":
        return "bfloat16" if torch_mod.cuda.is_bf16_supported() else "float16"
    if requested == "bfloat16" and not torch_mod.cuda.is_bf16_supported():
        raise BuildError(
            "compute-dtype=bfloat16 requested, but current GPU does not report BF16 support"
        )
    return requested



def resolve_compute_dtype(torch_mod: object, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch_mod.bfloat16
    if dtype_name == "float16":
        return torch_mod.float16
    raise BuildError(f"Unsupported dtype name: {dtype_name}")



def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise BuildError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)



def download_snapshot(repo_id: str, cache_dir: str | None, allow_patterns: List[str] | None = None) -> Path:
    print(f"[download] repo={repo_id}")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    path = Path(snapshot_path)
    print(f"[download] path={path}")
    return path



def validate_source_snapshot(source_dir: Path) -> None:
    missing = [f for f in REQUIRED_SOURCE_FILES if not (source_dir / f).exists()]
    if missing:
        raise BuildError(
            "Source repo is missing required files:\n" + "\n".join(f"- {m}" for m in missing)
        )



def copytree_clean(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )



def stage_runtime_files(runtime_dir: Path, output_dir: Path, template_loader: Path) -> None:
    print("[stage] copying runtime scaffold")
    wan_src = runtime_dir / "wan"
    generate_src = runtime_dir / "generate_prequant.py"
    requirements_src = runtime_dir / "requirements.txt"

    if not wan_src.exists() or not generate_src.exists() or not requirements_src.exists():
        raise BuildError(
            f"Runtime scaffold repo at {runtime_dir} is missing one or more required files"
        )

    copytree_clean(wan_src, output_dir / "wan")
    shutil.copy2(requirements_src, output_dir / "requirements.txt")
    shutil.copy2(generate_src, output_dir / "generate_prequant.py")
    shutil.copy2(template_loader, output_dir / "load_prequant.py")

    patch_generate_prequant(output_dir / "generate_prequant.py")
    make_executable(output_dir / "generate_prequant.py")
    make_executable(output_dir / "load_prequant.py")



def patch_generate_prequant(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = [
        ("pre-quantized NF4", "pre-quantized FP4"),
        ("Pre-quantized NF4", "Pre-quantized FP4"),
        ("quantized NF4", "quantized FP4"),
        ("_bnb_nf4", "_bnb_fp4"),
        ("bnb_nf4", "bnb_fp4"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    path.write_text(text, encoding="utf-8")



def stage_source_assets(source_dir: Path, output_dir: Path) -> None:
    print("[stage] copying tokenizer + T5 + VAE assets")
    copytree_clean(source_dir / "google" / "umt5-xxl", output_dir / "tokenizer")
    shutil.copy2(source_dir / "models_t5_umt5-xxl-enc-bf16.pth", output_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    shutil.copy2(source_dir / "Wan2.1_VAE.pth", output_dir / "Wan2.1_VAE.pth")



def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)



def import_wan_model(output_dir: Path):
    output_path = str(output_dir)
    if output_path not in sys.path:
        sys.path.insert(0, output_path)
    module = importlib.import_module("wan.modules.model")
    return module.WanModel



def quantize_submodel(
    *,
    torch_mod: object,
    bnb_mod: object,
    wan_model_cls: object,
    source_dir: Path,
    output_dir: Path,
    source_repo: str,
    source_sha: str,
    subfolder: str,
    quant_type: str,
    compute_dtype_name: str,
    compute_dtype,
    device: str,
    double_quant: bool,
) -> None:
    from safetensors.torch import save_file

    print(f"[quantize] subfolder={subfolder} quant_type={quant_type}")

    source_config_path = source_dir / subfolder / "config.json"
    if not source_config_path.exists():
        raise BuildError(f"Missing source config: {source_config_path}")
    with open(source_config_path, "r", encoding="utf-8") as f:
        source_config = json.load(f)

    load_dtype = compute_dtype
    model = wan_model_cls.from_pretrained(
        str(source_dir),
        subfolder=subfolder,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)

    linear_modules: List[Tuple[str, object]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch_mod.nn.Linear)
    ]
    if not linear_modules:
        raise BuildError(f"No nn.Linear layers found in {subfolder}; cannot quantize")

    linear_param_keys = set()
    for name, module in linear_modules:
        linear_param_keys.add(f"{name}.weight")
        if module.bias is not None:
            linear_param_keys.add(f"{name}.bias")

    quantized_state: Dict[str, object] = {}
    for key, tensor in model.state_dict().items():
        if key in linear_param_keys:
            continue
        quantized_state[key] = tensor.detach().cpu().contiguous()

    total = len(linear_modules)
    for idx, (name, module) in enumerate(linear_modules, start=1):
        print(f"[quantize]   layer {idx}/{total}: {name}")

        quant_linear = bnb_mod.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=double_quant,
            quant_type=quant_type,
        )

        fp_state = module.state_dict()
        quant_linear.load_state_dict(fp_state)
        quant_linear = quant_linear.to(device)

        for q_key, q_tensor in quant_linear.state_dict().items():
            quantized_state[f"{name}.{q_key}"] = q_tensor.detach().cpu().contiguous()

        del quant_linear
        del fp_state
        torch_mod.cuda.empty_cache()

    output_subdir = output_dir / f"{subfolder}_bnb_{quant_type}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    weights_path = output_subdir / "model.safetensors"
    save_file(quantized_state, str(weights_path))

    config_payload = dict(source_config)
    config_payload["_diffusers_version"] = importlib.import_module("diffusers").__version__
    with open(output_subdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
        f.write("\n")

    metadata = {
        "source": {
            "repo": source_repo,
            "sha": source_sha,
            "subfolder": subfolder,
        },
        "model_config": source_config,
        "quant": {
            "format": f"bnb_{quant_type}",
            "quant_type": quant_type,
            "double_quant": bool(double_quant),
            "compute_dtype": compute_dtype_name,
            "blocksize": 64,
        },
        "files": {
            "weights": "model.safetensors",
            "config": "config.json",
        },
        "stats": {
            "linear_layers_quantized": total,
            "quantized_bytes": weights_path.stat().st_size,
        },
        "tool_versions": {
            "python": platform.python_version(),
            "torch": importlib.import_module("torch").__version__,
            "bitsandbytes": importlib.import_module("bitsandbytes").__version__,
            "diffusers": importlib.import_module("diffusers").__version__,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_subdir / "quantization_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    del quantized_state
    del model
    gc.collect()
    torch_mod.cuda.empty_cache()



def generate_readme(
    output_dir: Path,
    source_repo: str,
    quant_type: str,
    compute_dtype_name: str,
    double_quant: bool,
) -> None:
    high_path = output_dir / f"high_noise_model_bnb_{quant_type}" / "model.safetensors"
    low_path = output_dir / f"low_noise_model_bnb_{quant_type}" / "model.safetensors"

    def fmt_gb(path: Path) -> str:
        return f"~{path.stat().st_size / (1024 ** 3):.1f}GB" if path.exists() else "(pending)"

    readme = f"""---
license: apache-2.0
library_name: pytorch
tags:
  - video-generation
  - image-to-video
  - diffusion
  - quantized
  - {quant_type}
  - bitsandbytes
pipeline_tag: image-to-video
---

# LingBot-World ACT {quant_type.upper()} Quantized

Pre-quantized {quant_type.upper()} weights for `{source_repo}`. This package is self-contained for inference.

## Quick Start

```bash
pip install -r requirements.txt

python generate_prequant.py \\
    --image your_image.jpg \\
    --prompt "A cinematic video of the scene" \\
    --frame_num 81 \\
    --size "480*832" \\
    --output output.mp4
```

## Model Contents

| File | Size | Description |
|------|------|-------------|
| `high_noise_model_bnb_{quant_type}/model.safetensors` | {fmt_gb(high_path)} | {quant_type.upper()} quantized diffusion model (high noise) |
| `low_noise_model_bnb_{quant_type}/model.safetensors` | {fmt_gb(low_path)} | {quant_type.upper()} quantized diffusion model (low noise) |
| `models_t5_umt5-xxl-enc-bf16.pth` | full precision | T5-XXL text encoder |
| `Wan2.1_VAE.pth` | full precision | VAE encoder/decoder |

## Quantization Details

```json
{{
  "format": "bnb_{quant_type}",
  "quant_type": "{quant_type}",
  "double_quant": {str(bool(double_quant)).lower()},
  "compute_dtype": "{compute_dtype_name}",
  "blocksize": 64
}}
```

## Requirements

- Python 3.10+
- CUDA 11.8+ (CUDA 12.x recommended)
- bitsandbytes-compatible NVIDIA GPU
- High-memory GPU for conversion (A100/H100 recommended)

## Caveat

This quantized model is intended for inference. Minor degradation in visual fidelity or temporal consistency can occur versus full precision.
"""

    (output_dir / "README.md").write_text(readme, encoding="utf-8")



def fetch_repo_sha(repo_id: str) -> str:
    try:
        info = HfApi().model_info(repo_id)
        return info.sha or "unknown"
    except Exception:
        return "unknown"



def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    template_loader = Path(__file__).resolve().parent / "templates" / "load_prequant.py"
    if not template_loader.exists():
        raise BuildError(f"Loader template not found: {template_loader}")

    torch_mod, bnb_mod, _diffusers_mod, compute_dtype_name = preflight_environment(args)
    compute_dtype = resolve_compute_dtype(torch_mod, compute_dtype_name)

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    source_dir = download_snapshot(args.source_repo, cache_dir=args.cache_dir)
    validate_source_snapshot(source_dir)

    runtime_dir = download_snapshot(
        args.runtime_scaffold_repo,
        cache_dir=args.cache_dir,
        allow_patterns=RUNTIME_ALLOW_PATTERNS,
    )

    stage_runtime_files(runtime_dir, output_dir, template_loader)
    stage_source_assets(source_dir, output_dir)

    source_sha = fetch_repo_sha(args.source_repo)
    wan_model_cls = import_wan_model(output_dir)

    quantize_submodel(
        torch_mod=torch_mod,
        bnb_mod=bnb_mod,
        wan_model_cls=wan_model_cls,
        source_dir=source_dir,
        output_dir=output_dir,
        source_repo=args.source_repo,
        source_sha=source_sha,
        subfolder="high_noise_model",
        quant_type="fp4",
        compute_dtype_name=compute_dtype_name,
        compute_dtype=compute_dtype,
        device=args.device,
        double_quant=args.double_quant,
    )

    quantize_submodel(
        torch_mod=torch_mod,
        bnb_mod=bnb_mod,
        wan_model_cls=wan_model_cls,
        source_dir=source_dir,
        output_dir=output_dir,
        source_repo=args.source_repo,
        source_sha=source_sha,
        subfolder="low_noise_model",
        quant_type="fp4",
        compute_dtype_name=compute_dtype_name,
        compute_dtype=compute_dtype,
        device=args.device,
        double_quant=args.double_quant,
    )

    generate_readme(
        output_dir=output_dir,
        source_repo=args.source_repo,
        quant_type="fp4",
        compute_dtype_name=compute_dtype_name,
        double_quant=args.double_quant,
    )

    print("[done] FP4 package build complete")
    print(f"[done] output: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)
