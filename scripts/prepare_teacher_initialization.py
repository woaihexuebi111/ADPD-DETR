"""Transfer compatible OWRT COCO tensors into the current P2 teacher."""
from __future__ import annotations
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-config"))
from ultralytics import RTDETR, __version__

# Source OWRT layer -> target four-scale model layer.
#
# These mappings are intentionally conservative. They copy layers whose tensor
# shapes and roles are close enough to reuse. Newly introduced four-scale paths
# and DGFE-specific parameters remain target-initialized.
PREFIX_MAPPINGS = [
    ("model.8.", "model.9.", "P5 projection"),
    ("model.9.", "model.10.", "AIFI on P5"),
    ("model.10.", "model.11.", "P5 post projection"),
    ("model.14.", "model.15.", "stride-16 RetBlock fusion seed"),
    ("model.15.", "model.16.", "stride-16 post Conv seed"),
    ("model.19.", "model.20.", "stride-8 RetBlock fusion seed"),
    ("model.17.", "model.23.", "P2 lateral projection seed"),
    ("model.19.", "model.25.", "P2 top-down RetBlock seed"),
    ("model.24.", "model.26.", "P2->P3 PAN downsample seed"),
    ("model.24.", "model.29.", "P3->P4 PAN downsample seed"),
    ("model.27.", "model.32.", "P4->P5 PAN downsample seed"),
    ("model.26.", "model.28.", "P3 PAN RetBlock seed"),
    ("model.26.", "model.31.", "P4 PAN RetBlock seed"),
    ("model.29.", "model.34.", "P5 PAN RetBlock seed"),
]


def _tensor_numel(state: dict[str, torch.Tensor], keys: set[str]) -> int:
    return int(sum(state[key].numel() for key in keys))


def _add_copy(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    copy_plan: dict[str, dict],
    source_key: str,
    target_key: str,
    reason: str,
    overwrite: bool = True,
) -> bool:
    if source_key not in source_state or target_key not in target_state:
        return False
    if tuple(source_state[source_key].shape) != tuple(target_state[target_key].shape):
        return False
    if target_key in copy_plan and not overwrite:
        return False
    copy_plan[target_key] = {
        "source": source_key,
        "reason": reason,
        "shape": list(source_state[source_key].shape),
    }
    return True


def _copy_prefix(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    copy_plan: dict[str, dict],
    source_prefix: str,
    target_prefix: str,
    reason: str,
) -> int:
    count = 0
    for source_key in source_state:
        if not source_key.startswith(source_prefix):
            continue
        target_key = f"{target_prefix}{source_key[len(source_prefix):]}"
        if _add_copy(source_state, target_state, copy_plan, source_key, target_key, reason):
            count += 1
    return count


def _copy_decoder(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    copy_plan: dict[str, dict],
) -> int:
    """Copy compatible 3-level OWRT decoder weights into the 4-level decoder.

    Cross-attention sampling-offset and attention-weight tensors have different
    first dimensions because the new decoder has 4 feature levels instead of 3;
    those are left initialized by the target YAML.
    """

    count = 0

    # Duplicate the old lowest-level input projection for the new P2 slot, then
    # shift the original three projections to P3/P4/P5 slots.
    input_projection_map = [
        ("model.30.input_proj.0.", "model.35.input_proj.0."),
        ("model.30.input_proj.0.", "model.35.input_proj.1."),
        ("model.30.input_proj.1.", "model.35.input_proj.2."),
        ("model.30.input_proj.2.", "model.35.input_proj.3."),
    ]
    for source_prefix, target_prefix in input_projection_map:
        count += _copy_prefix(
            source_state,
            target_state,
            copy_plan,
            source_prefix,
            target_prefix,
            "decoder input projection seed",
        )

    skip_fragments = (
        ".input_proj.",
        ".cross_attn.sampling_offsets.",
        ".cross_attn.attention_weights.",
    )
    for source_key in source_state:
        if not source_key.startswith("model.30."):
            continue
        if any(fragment in source_key for fragment in skip_fragments):
            continue
        target_key = f"model.35.{source_key[len('model.30.'):]}"
        if _add_copy(
            source_state,
            target_state,
            copy_plan,
            source_key,
            target_key,
            "decoder compatible shared tensor",
        ):
            count += 1
    return count


def prepare_one(source_weights: Path, yaml_path: Path, output: Path, label: str, seed: int, overwrite: bool) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    torch.manual_seed(seed)
    source_model = RTDETR(str(source_weights)).model.float().cpu()
    target_wrapper = RTDETR(str(yaml_path))
    target_model = target_wrapper.model.float().cpu()

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    copy_plan: dict[str, dict] = {}

    # Direct same-key copy mostly covers ResNet stem/backbone layers 0-7.
    direct_count = 0
    for source_key in source_state:
        if _add_copy(
            source_state,
            target_state,
            copy_plan,
            source_key,
            source_key,
            "same-key compatible tensor",
            overwrite=False,
        ):
            direct_count += 1

    prefix_counts = {}
    for source_prefix, target_prefix, reason in PREFIX_MAPPINGS:
        prefix_counts[f"{source_prefix}->{target_prefix}"] = _copy_prefix(
            source_state,
            target_state,
            copy_plan,
            source_prefix,
            target_prefix,
            reason,
        )

    decoder_count = _copy_decoder(source_state, target_state, copy_plan)

    new_state = target_model.state_dict()
    for target_key, item in copy_plan.items():
        new_state[target_key] = source_state[item["source"]].detach().clone()
    target_model.load_state_dict(new_state, strict=True)

    target_model.args = {
        "task": "detect",
        "model": str(yaml_path),
        "pretrained": True,
    }
    target_model.task = "detect"

    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(target_model).float(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": target_model.args,
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().isoformat(),
        "version": __version__,
        "source": f"{label}: fair OWRT initialization with target-specific DGFE tensors",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    reloaded = RTDETR(str(output)).model.float().cpu()
    reloaded_state = reloaded.state_dict()
    max_reload_difference = 0.0
    for target_key, item in copy_plan.items():
        max_reload_difference = max(
            max_reload_difference,
            float((reloaded_state[target_key] - source_state[item["source"]]).abs().max()),
        )
    if max_reload_difference != 0.0:
        raise RuntimeError(f"Reload verification failed: {max_reload_difference}")

    copied_keys = set(copy_plan)
    new_keys = set(target_state) - copied_keys
    report = {
        "label": label,
        "source_weights": str(source_weights),
        "yaml": str(yaml_path),
        "output_weights": str(output),
        "seed": seed,
        "target_tensors": len(target_state),
        "copied_tensors": len(copied_keys),
        "new_or_shape_changed_tensors": len(new_keys),
        "copied_numel": _tensor_numel(target_state, copied_keys),
        "new_or_shape_changed_numel": _tensor_numel(target_state, new_keys),
        "direct_same_key_copies": direct_count,
        "prefix_copy_counts": prefix_counts,
        "decoder_copies": decoder_count,
        "max_shared_reload_difference": max_reload_difference,
        "created": datetime.now().isoformat(),
        "notes": [
            "DGFE branch tensors are intentionally target-initialized.",
            "Four-level decoder cross-attention offset/weight tensors are target-initialized because their shapes differ from 3-level OWRT.",
        ],
    }
    output.with_suffix(".preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output.with_suffix(".copy_plan.json").write_text(
        json.dumps(copy_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report

