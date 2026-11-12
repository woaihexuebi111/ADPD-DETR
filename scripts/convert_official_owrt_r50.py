"""Create a reproducible OWRT-DETR-R50 checkpoint from official RT-DETR weights.

OWRT keeps the RT-DETR backbone, AIFI, several projection/downsample layers,
and the Transformer decoder, while replacing the FPN/PAN fusion blocks and
adding CFPIM/MSSF/SOEM modules. Shared tensors are copied exactly; OWRT-only
parameters are initialized deterministically.
"""

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

from convert_official_rtdetr_r50 import (
    BACKBONE_STAGES,
    BACKBONE_STEM,
    load_official_state,
    map_decoder_key,
    replace_prefix,
)
from ultralytics import RTDETR, __version__


OWRT_NEW_PREFIXES = (
    "model.12.",
    "model.14.",
    "model.17.",
    "model.19.",
    "model.20.",
    "model.21.",
    "model.23.",
    "model.26.",
    "model.29.",
)


def map_owrt_encoder_key(key: str) -> str | None:
    """Map official HybridEncoder tensors that OWRT preserves structurally."""
    direct = {
        "encoder.input_proj.2": "model.8",
        "encoder.lateral_convs.0": "model.10",
        "encoder.lateral_convs.1": "model.15",
        "encoder.downsample_convs.0": "model.24",
        "encoder.downsample_convs.1": "model.27",
    }
    mapped = replace_prefix(key, direct)
    if mapped is not None:
        mapped = mapped.replace(".norm.", ".bn.")
        for layer in (8, 12, 17):
            mapped = mapped.replace(f"model.{layer}.0.", f"model.{layer}.conv.")
            mapped = mapped.replace(f"model.{layer}.1.", f"model.{layer}.bn.")
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


def map_owrt_key(key: str, decoder_index: int) -> str | None:
    mapped = replace_prefix(key, BACKBONE_STEM)
    if mapped is None:
        mapped = replace_prefix(key, BACKBONE_STAGES)
    if mapped is None:
        mapped = map_owrt_encoder_key(key)
    if mapped is None:
        mapped = map_decoder_key(key, decoder_index=decoder_index)
    return mapped


def convert(
    official_path: Path,
    yaml_path: Path,
    output_path: Path,
    seed: int,
) -> dict:
    official = load_official_state(official_path)
    torch.manual_seed(seed)
    wrapper = RTDETR(str(yaml_path))
    model = wrapper.model.float()
    target_state = model.state_dict()
    decoder_index = len(model.model) - 1

    converted = {}
    source_to_target = {}
    skipped_source = []
    errors = []
    for source_key, tensor in official.items():
        target_key = map_owrt_key(source_key, decoder_index)
        if target_key is None:
            skipped_source.append(source_key)
            continue
        if target_key not in target_state:
            errors.append(
                {"source": source_key, "target": target_key, "error": "missing target key"}
            )
            continue
        if tuple(tensor.shape) != tuple(target_state[target_key].shape):
            errors.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[target_key].shape),
                }
            )
            continue
        converted[target_key] = tensor
        source_to_target[source_key] = target_key
    if errors:
        raise RuntimeError(json.dumps(errors[:30], indent=2))

    model.load_state_dict(converted, strict=False)
    unmapped_target = sorted(set(target_state) - set(converted))
    allowed_unmapped = [
        key
        for key in unmapped_target
        if key.endswith("num_batches_tracked")
        or key.startswith(OWRT_NEW_PREFIXES)
    ]
    disallowed_unmapped = sorted(set(unmapped_target) - set(allowed_unmapped))
    if disallowed_unmapped:
        raise RuntimeError(
            json.dumps(
                {
                    "disallowed_unmapped": disallowed_unmapped,
                    "allowed_unmapped": allowed_unmapped,
                },
                indent=2,
            )
        )

    trainable = dict(model.named_parameters())
    mapped_trainable = {key for key in converted if key in trainable}
    shared_trainable = {
        key for key in trainable if not key.startswith(OWRT_NEW_PREFIXES)
    }
    missing_shared = sorted(shared_trainable - mapped_trainable)
    if missing_shared:
        raise RuntimeError(f"Unmapped shared trainable tensors: {missing_shared[:30]}")

    model.args = {"task": "detect", "model": str(yaml_path), "pretrained": True}
    model.task = "detect"
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).float(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": model.args,
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().isoformat(),
        "version": __version__,
        "source": "Official RT-DETR-R50 plus deterministic OWRT initialization",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    report = {
        "yaml": str(yaml_path),
        "output": str(output_path),
        "seed": seed,
        "official_tensors": len(official),
        "mapped_official_tensors": len(converted),
        "skipped_replaced_encoder_tensors": len(skipped_source),
        "target_tensors": len(target_state),
        "unmapped_target_tensors": len(unmapped_target),
        "mapped_shared_trainable_tensors": len(shared_trainable & mapped_trainable),
        "shared_trainable_tensors": len(shared_trainable),
        "mapped_shared_trainable_numel": sum(
            trainable[key].numel() for key in shared_trainable & mapped_trainable
        ),
        "shared_trainable_numel": sum(
            trainable[key].numel() for key in shared_trainable
        ),
        "owrt_new_trainable_numel": sum(
            parameter.numel()
            for key, parameter in trainable.items()
            if key.startswith(OWRT_NEW_PREFIXES)
        ),
        "allowed_uninitialized_keys": allowed_unmapped,
        "skipped_source_keys": skipped_source,
        "source_to_target": source_to_target,
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("weights/rtdetr_r50vd_6x_coco_official.pth"),
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("ultralytics/cfg/models/OWRT/OWRT.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/owrt_detr_r50_fair_coco_ultralytics.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = convert(args.official, args.yaml, args.output, args.seed)
    print(
        json.dumps(
            {key: value for key, value in report.items() if not isinstance(value, list) and key != "source_to_target"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
