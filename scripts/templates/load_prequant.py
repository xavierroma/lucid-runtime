#!/usr/bin/env python3
"""
Load pre-quantized bitsandbytes 4-bit WanModel checkpoints.

Supports both NF4 and FP4 checkpoints with dynamic quant-state key detection.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
from bitsandbytes.functional import QuantState

# Allow direct execution from package root.
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.model import WanModel


def replace_linears_with_bnb_4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
    quant_type: str = "nf4",
) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """Replace all nn.Linear modules with bnb Linear4bit placeholders."""
    replaced = 0
    layer_shapes: Dict[str, Tuple[int, int]] = {}

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for name, module in linear_layers:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        layer_shapes[name] = (module.in_features, module.out_features)

        quant_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        setattr(parent, child_name, quant_linear)
        replaced += 1

    return replaced, layer_shapes



def build_model_from_config(config: Dict[str, Any]) -> WanModel:
    """Build a WanModel from a config dict."""
    return WanModel(
        model_type=config.get("model_type", "i2v"),
        patch_size=tuple(config.get("patch_size", (1, 2, 2))),
        text_len=config.get("text_len", 512),
        in_dim=config.get("in_dim", 16),
        dim=config.get("dim", 2048),
        ffn_dim=config.get("ffn_dim", 8192),
        freq_dim=config.get("freq_dim", 256),
        text_dim=config.get("text_dim", 4096),
        out_dim=config.get("out_dim", 16),
        num_heads=config.get("num_heads", 16),
        num_layers=config.get("num_layers", 32),
        window_size=tuple(config.get("window_size", (-1, -1))),
        qk_norm=config.get("qk_norm", True),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6),
    )



def _pick_quant_state_key(
    weight_components: Dict[str, torch.Tensor], expected_quant_type: str
) -> str:
    quant_keys = [k for k in weight_components if k.startswith("quant_state.bitsandbytes__")]
    if quant_keys:
        return quant_keys[0]

    fallback = f"quant_state.bitsandbytes__{expected_quant_type}"
    if fallback in weight_components:
        return fallback

    raise KeyError(
        "No bitsandbytes quant_state key found for layer. "
        f"Available component keys: {sorted(weight_components.keys())}"
    )



def reconstruct_params4bit_from_components(
    weight_components: Dict[str, torch.Tensor],
    device: str,
    quant_state_key: str,
) -> bnb.nn.Params4bit:
    """Reconstruct a Params4bit tensor from serialized components."""
    qs_dict: Dict[str, torch.Tensor] = {
        "absmax": weight_components["absmax"],
        "quant_map": weight_components["quant_map"],
    }

    if "nested_absmax" in weight_components:
        qs_dict["nested_absmax"] = weight_components["nested_absmax"]
        qs_dict["nested_quant_map"] = weight_components["nested_quant_map"]

    qs_dict[quant_state_key] = weight_components[quant_state_key]
    quant_state = QuantState.from_dict(qs_dict, device=torch.device(device))

    quantized_weight = weight_components["weight"].to(device)
    return bnb.nn.Params4bit(
        data=quantized_weight,
        requires_grad=False,
        quant_state=quant_state,
        bnb_quantized=True,
    )



def _parse_quantized_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[
    Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
]:
    """Split a state_dict into quantized linear components and other tensors."""
    weight_components: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    pending: Dict[str, torch.Tensor] = {}
    quantized_bases = set()

    for key, tensor in sd.items():
        base_key = None
        component = None

        if ".weight.absmax" in key:
            base_key = key.replace(".weight.absmax", "")
            component = "absmax"
        elif ".weight.quant_map" in key:
            base_key = key.replace(".weight.quant_map", "")
            component = "quant_map"
        elif ".weight.nested_absmax" in key:
            base_key = key.replace(".weight.nested_absmax", "")
            component = "nested_absmax"
        elif ".weight.nested_quant_map" in key:
            base_key = key.replace(".weight.nested_quant_map", "")
            component = "nested_quant_map"
        elif ".weight.quant_state.bitsandbytes__" in key:
            base_key, suffix = key.split(".weight.", 1)
            component = suffix

        if base_key is not None and component is not None:
            weight_components[base_key][component] = tensor
            quantized_bases.add(base_key)
        else:
            pending[key] = tensor

    other_keys: Dict[str, torch.Tensor] = {}
    for key, tensor in pending.items():
        if key.endswith(".weight") and key[:-7] in quantized_bases:
            weight_components[key[:-7]]["weight"] = tensor
        else:
            other_keys[key] = tensor

    return weight_components, other_keys



def load_quantized_state(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str = "cpu",
    expected_quant_type: str = "nf4",
) -> nn.Module:
    """Load quantized 4-bit weights into a model with Linear4bit modules."""
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(weights_path)
    else:
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)

    weight_components, other_keys = _parse_quantized_state_dict(sd)

    loaded_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue

        components = weight_components.get(name)
        if components is None or "weight" not in components:
            continue

        quant_state_key = _pick_quant_state_key(components, expected_quant_type)
        module.weight = reconstruct_params4bit_from_components(
            components, device=device, quant_state_key=quant_state_key
        )
        loaded_count += 1

        bias_key = f"{name}.bias"
        if bias_key in other_keys and module.bias is not None:
            module.bias.data.copy_(other_keys[bias_key].to(device))

    non_linear_sd = dict(other_keys)
    if non_linear_sd:
        missing, _unexpected = model.load_state_dict(non_linear_sd, strict=False)

        expected_missing = {f"{name}.weight" for name in layer_shapes}
        expected_missing.update({f"{name}.bias" for name in layer_shapes})
        critical_missing = [
            k for k in missing if k not in expected_missing and not k.endswith("freqs")
        ]
        if critical_missing:
            print(f"Warning: Missing non-quantized keys: {critical_missing[:10]}...")

    print(f"  Loaded {loaded_count} quantized linear layers")
    return model



def load_quantized_model(
    model_dir: str,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> WanModel:
    """Load a pre-quantized WanModel directory (FP4 or NF4)."""
    model_dir_path = Path(model_dir)

    config_path = model_dir_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    quant_type = "nf4"
    compress_statistics = True

    meta_path = model_dir_path / "quantization_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        quant_cfg = meta.get("quant", {})
        quant_type = quant_cfg.get("quant_type", quant_type)
        compress_statistics = bool(quant_cfg.get("double_quant", compress_statistics))
        compute_dtype_name = quant_cfg.get("compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)

    safetensors_path = model_dir_path / "model.safetensors"
    pt_path = model_dir_path / "model.pt"
    if safetensors_path.exists():
        weights_path = str(safetensors_path)
    elif pt_path.exists():
        weights_path = str(pt_path)
    else:
        raise FileNotFoundError(
            f"No weights found in {model_dir_path}. "
            "Expected model.safetensors or model.pt"
        )

    print(f"Loading pre-quantized model from {model_dir_path}")
    print(f"  Config: {config_path}")
    print(f"  Weights: {weights_path}")
    print(f"  Quant type: {quant_type}")

    model = build_model_from_config(config)
    replaced, layer_shapes = replace_linears_with_bnb_4bit(
        model,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    print(f"  Replaced {replaced} linear layers with bnb.Linear4bit")

    model = load_quantized_state(
        model,
        weights_path,
        layer_shapes,
        device=device,
        expected_quant_type=quant_type,
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"  Model ready on {device}")
    return model



def verify_quantized_model(model: nn.Module) -> Dict[str, Any]:
    """Return basic verification metrics for a quantized model."""
    total_params = 0
    quantized_params = 0
    linear4bit_count = 0
    regular_linear_count = 0

    for _name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear4bit_count += 1
            if hasattr(module.weight, "quant_state") and module.weight.quant_state is not None:
                quantized_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            regular_linear_count += 1

    for param in model.parameters():
        total_params += param.numel()

    return {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "linear4bit_count": linear4bit_count,
        "regular_linear_count": regular_linear_count,
        "is_quantized": linear4bit_count > 0 and regular_linear_count == 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test loading pre-quantized model")
    parser.add_argument("model_dir", type=str, help="Path to quantized model directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load to")
    args = parser.parse_args()

    model = load_quantized_model(args.model_dir, device=args.device)
    info = verify_quantized_model(model)

    print("\nVerification:")
    print(f"  Linear4bit layers: {info['linear4bit_count']}")
    print(f"  Regular Linear layers: {info['regular_linear_count']}")
    print(f"  Is properly quantized: {info['is_quantized']}")
