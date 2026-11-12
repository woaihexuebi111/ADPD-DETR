"""Convert the official RT-DETR R50 checkpoint to this local Ultralytics layout."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from ultralytics import RTDETR, __version__
from ultralytics.nn.modules import DifferenceMapGuidedEnhancement


BACKBONE_STEM = {
    "backbone.conv1.conv1_1": "model.0",
    "backbone.conv1.conv1_2": "model.1",
    "backbone.conv1.conv1_3": "model.2",
}
BACKBONE_STAGES = {
    "backbone.res_layers.0": "model.4",
    "backbone.res_layers.1": "model.5",
    "backbone.res_layers.2": "model.6",
    "backbone.res_layers.3": "model.7",
}


def load_official_state(path: Path) -> dict[str, torch.Tensor]:
    """Return the EMA module state from the official RT-DETR checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        state = checkpoint["ema"]["module"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unsupported official checkpoint structure: {path}") from exc
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict, got {type(state)!r}")
    return state


def replace_prefix(key: str, mapping: dict[str, str]) -> str | None:
    """Replace the first matching prefix in a state-dict key."""
    for source, target in mapping.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return None


def map_encoder_key(key: str, dgfe: bool) -> str | None:
    """Map an official HybridEncoder key to the flattened Ultralytics neck."""
    downsample_layers = (21, 24) if dgfe else (20, 23)
    pan_layers = (23, 26) if dgfe else (22, 25)

    direct = {
        "encoder.input_proj.2": "model.8",
        "encoder.lateral_convs.0": "model.10",
        "encoder.input_proj.1": "model.12",
        "encoder.fpn_blocks.0": "model.14",
        "encoder.lateral_convs.1": "model.15",
        "encoder.input_proj.0": "model.17",
        "encoder.fpn_blocks.1": "model.19",
        "encoder.downsample_convs.0": f"model.{downsample_layers[0]}",
        "encoder.pan_blocks.0": f"model.{pan_layers[0]}",
        "encoder.downsample_convs.1": f"model.{downsample_layers[1]}",
        "encoder.pan_blocks.1": f"model.{pan_layers[1]}",
    }
    mapped = replace_prefix(key, direct)
    if mapped is not None:
        # Official ConvNormLayer uses "norm"; local Conv/RepConv uses "bn".
        mapped = mapped.replace(".norm.", ".bn.")
        # Official nn.Sequential input projections become local Conv modules.
        for layer in (8, 12, 17):
            mapped = mapped.replace(f"model.{layer}.0.", f"model.{layer}.conv.")
            mapped = mapped.replace(f"model.{layer}.1.", f"model.{layer}.bn.")
        # Official CSPRepLayer names versus local RepC3 names.
        mapped = mapped.replace(".conv1.", ".cv1.", 1)
        mapped = mapped.replace(".conv2.", ".cv2.", 1)
        mapped = mapped.replace(".bottlenecks.", ".m.")
        # Undo the outer rename inside each RepVgg/RepConv block.
        mapped = mapped.replace(".m.", ".m.", 1)
        mapped = mapped.replace(".m.0.cv1.", ".m.0.conv1.")
        mapped = mapped.replace(".m.0.cv2.", ".m.0.conv2.")
        mapped = mapped.replace(".m.1.cv1.", ".m.1.conv1.")
        mapped = mapped.replace(".m.1.cv2.", ".m.1.conv2.")
        mapped = mapped.replace(".m.2.cv1.", ".m.2.conv1.")
        mapped = mapped.replace(".m.2.cv2.", ".m.2.conv2.")
        return mapped

    encoder_prefix = "encoder.encoder.0.layers.0"
    if key == encoder_prefix or key.startswith(encoder_prefix + "."):
        mapped = "model.9" + key[len(encoder_prefix) :]
        return (
            mapped.replace(".self_attn.", ".ma.")
            .replace(".linear1.", ".fc1.")
            .replace(".linear2.", ".fc2.")
        )
    return None


def map_decoder_key(key: str, decoder_index: int) -> str | None:
    """Map an official decoder key to the local RTDETRDecoder module."""
    if not key.startswith("decoder."):
        return None
    mapped = f"model.{decoder_index}." + key[len("decoder.") :]
    for index in range(3):
        mapped = mapped.replace(
            f"model.{decoder_index}.input_proj.{index}.conv.",
            f"model.{decoder_index}.input_proj.{index}.0.",
        )
        mapped = mapped.replace(
            f"model.{decoder_index}.input_proj.{index}.norm.",
            f"model.{decoder_index}.input_proj.{index}.1.",
        )
    return mapped


def map_key(key: str, dgfe: bool, decoder_index: int) -> str:
    """Map one official checkpoint key and fail loudly on unknown structure."""
    mapped = replace_prefix(key, BACKBONE_STEM)
    if mapped is None:
        mapped = replace_prefix(key, BACKBONE_STAGES)
    if mapped is None:
        mapped = map_encoder_key(key, dgfe=dgfe)
    if mapped is None:
        mapped = map_decoder_key(key, decoder_index=decoder_index)
    if mapped is None:
        raise KeyError(f"No mapping rule for official key: {key}")
    return mapped


def convert_one(
    official: dict[str, torch.Tensor],
    yaml_path: Path,
    output_path: Path,
    initialization_seed: int = 42,
) -> dict:
    """Build, populate, verify and save one local RT-DETR checkpoint."""
    # Reset before every model build so controlled DGFE variants receive
    # identical newly initialized reconstruction/attention parameters.
    torch.manual_seed(initialization_seed)
    wrapper = RTDETR(str(yaml_path))
    model = wrapper.model.float()
    target_state = model.state_dict()
    dgfe_modules = [module for module in model.modules() if isinstance(module, DifferenceMapGuidedEnhancement)]
    dgfe = bool(dgfe_modules)
    decoder_index = len(model.model) - 1

    converted: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    shape_errors: list[dict] = []
    for source_key, tensor in official.items():
        target_key = map_key(source_key, dgfe=dgfe, decoder_index=decoder_index)
        source_to_target[source_key] = target_key
        if target_key not in target_state:
            shape_errors.append({"source": source_key, "target": target_key, "error": "missing target key"})
            continue
        if tuple(tensor.shape) != tuple(target_state[target_key].shape):
            shape_errors.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[target_key].shape),
                }
            )
            continue
        if target_key in converted:
            raise RuntimeError(f"Duplicate mapped target key: {target_key}")
        converted[target_key] = tensor

    if shape_errors:
        raise RuntimeError(json.dumps(shape_errors[:20], indent=2))
    if len(converted) != len(official):
        raise RuntimeError(f"Only converted {len(converted)}/{len(official)} official tensors")

    incompatible = model.load_state_dict(converted, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    unmapped_target_keys = sorted(set(target_state) - set(converted))
    allowed_unmapped = [
        key
        for key in unmapped_target_keys
        if key.endswith("num_batches_tracked")
        or (dgfe and key.startswith("model.20."))
    ]
    disallowed_unmapped = sorted(set(unmapped_target_keys) - set(allowed_unmapped))
    if unexpected or disallowed_unmapped:
        raise RuntimeError(
            json.dumps(
                {
                    "unexpected": unexpected,
                    "load_state_missing": missing,
                    "disallowed_unmapped": disallowed_unmapped,
                    "allowed_unmapped": allowed_unmapped,
                },
                indent=2,
            )
        )

    trainable = {key: value for key, value in model.named_parameters()}
    matched_trainable = {key for key in converted if key in trainable}
    base_trainable = {
        key
        for key in trainable
        if not (dgfe and key.startswith("model.20."))
    }
    missing_base_trainable = sorted(base_trainable - matched_trainable)
    if missing_base_trainable:
        raise RuntimeError(f"Unmatched base trainable parameters: {missing_base_trainable[:20]}")

    model.args = {"task": "detect", "model": str(yaml_path), "pretrained": True}
    model.task = "detect"
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        # Keep the official FP32 tensors exact. Training checkpoints may later
        # be stored as FP16 by the trainer, but conversion should be lossless.
        "model": deepcopy(model).float(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": model.args,
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().isoformat(),
        "version": __version__,
        "source": "RT-DETR official rtdetr_r50vd_6x_coco_from_paddle.pth",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    report = {
        "yaml": str(yaml_path),
        "output": str(output_path),
        "dgfe": dgfe,
        "initialization_seed": initialization_seed,
        "official_tensors": len(official),
        "converted_tensors": len(converted),
        "target_tensors": len(target_state),
        "unmapped_target_tensors": len(unmapped_target_keys),
        "unmapped_batch_counters": sum(key.endswith("num_batches_tracked") for key in unmapped_target_keys),
        "unmapped_dgfe_tensors": sum(dgfe and key.startswith("model.20.") for key in unmapped_target_keys),
        "allowed_uninitialized_keys": allowed_unmapped,
        "base_trainable_parameters": len(base_trainable),
        "matched_base_trainable_parameters": len(base_trainable & matched_trainable),
        "base_trainable_numel": sum(trainable[key].numel() for key in base_trainable),
        "matched_base_trainable_numel": sum(trainable[key].numel() for key in base_trainable & matched_trainable),
        "dgfe_trainable_numel": sum(
            parameter.numel()
            for key, parameter in trainable.items()
            if dgfe and key.startswith("model.20.")
        ),
        "source_to_target": source_to_target,
    }
    report_path = output_path.with_suffix(".conversion.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("weights/rtdetr_r50vd_6x_coco_official.pth"),
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("YAML", "OUTPUT"),
        help="YAML and output .pt pair; may be supplied more than once.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = args.model or [
        (
            "ultralytics/cfg/models/RTDETR/rtdetr-ResNet50.yaml",
            "weights/rtdetr_r50_coco_ultralytics.pt",
        ),
        (
            "ultralytics/cfg/models/RTDETR/rtdetr-ResNet50-DGFE-Strict.yaml",
            "weights/rtdetr_r50_dgfe_strict_coco_ultralytics.pt",
        ),
        (
            "ultralytics/cfg/models/RTDETR/rtdetr-ResNet50-DGFE-Warmup.yaml",
            "weights/rtdetr_r50_dgfe_warmup_coco_ultralytics.pt",
        ),
    ]
    official = load_official_state(args.official)
    summaries = []
    for yaml_name, output_name in pairs:
        report = convert_one(
            official,
            Path(yaml_name),
            Path(output_name),
            initialization_seed=args.seed,
        )
        summaries.append({key: value for key, value in report.items() if key != "source_to_target"})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
