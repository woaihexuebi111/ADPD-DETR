"""Report measured parameter count and THOP GFLOPs for a YAML or checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-config"))

from ultralytics import RTDETR  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--unfused", action="store_true", help="Profile the unfused network instead of deployment fusion")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    wrapper = RTDETR(str(args.model))
    wrapper.model.eval()
    if not args.unfused:
        wrapper.fuse()
    network = wrapper.model
    result = {
        "model": str(args.model),
        "input_size": [args.imgsz, args.imgsz],
        "parameters_m": sum(p.numel() for p in network.parameters()) / 1e6,
        "gflops": float(get_flops(network, imgsz=args.imgsz)),
        "model_state": "unfused" if args.unfused else "fused",
        "classes": int(network.model[-1].nc),
        "note": "THOP-supported operator MACs multiplied by two, at the requested input dimensions.",
    }
    if result["gflops"] <= 0:
        raise RuntimeError("Profiling failed; a zero GFLOPs value is not a valid measurement")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
