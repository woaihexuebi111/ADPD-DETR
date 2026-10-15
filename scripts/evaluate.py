"""Evaluate a trusted student checkpoint using the standard COCO bbox protocol."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-config"))

from ultralytics import RTDETR
from ultralytics.models.rtdetr.coco import COCORTDETRValidator
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    model = RTDETR(str(args.weights))
    validator = COCORTDETRValidator(args=dict(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, save_txt=False, save_json=False,
        plots=False, verbose=False, mode="val", project=str(args.output / "runs"),
        name=f"{args.weights.stem}_{args.split}", exist_ok=False))
    validator(model=model.model)
    (args.output / f"{args.split}_ground_truth.json").write_text(
        json.dumps(validator.coco_ground_truth), encoding="utf-8")
    (args.output / f"{args.weights.stem}_{args.split}_predictions.json").write_text(
        json.dumps(validator.coco_predictions), encoding="utf-8")
    model.fuse()
    flops = float(get_flops(model.model, imgsz=args.imgsz))
    if flops <= 0:
        raise RuntimeError("GFLOPs profiling failed; zero is not a valid complexity measurement")
    result = {"weights": str(args.weights), "data": str(args.data), "split": args.split,
              "images": len(validator.coco_images), "imgsz": args.imgsz,
              "coco": validator.coco_metrics,
              "parameters_m": sum(p.numel() for p in model.model.parameters()) / 1e6,
              "gflops": flops, "profile_state": "fused inference model",
              "small_object_rule": "COCO annotation area in original image coordinates",
              "metric_scale": "fractions, not percentages"}
    output = args.output / f"{args.weights.stem}_{args.split}_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
