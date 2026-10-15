"""Train the reconstruction-guided ADPD teacher model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-config"))

from ultralytics import RTDETR  # noqa: E402


DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/ADPD/ADPD-Teacher-R50.yaml"
DEFAULT_DATA = ROOT / "configs/datasets/complex_sea.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="adpd_teacher")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Missing teacher model configuration: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset configuration: {args.data}")
    if args.init_weights is not None and not args.init_weights.exists():
        raise FileNotFoundError(f"Missing initialization checkpoint: {args.init_weights}")

    model = RTDETR(str(args.model))
    if args.init_weights is not None:
        model.load(str(args.init_weights))

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        seed=args.seed,
        amp=args.amp,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        pretrained=False,
        lrf=1.0,
        warmup_epochs=2000,
        patience=0,
        plots=False,
    )


if __name__ == "__main__":
    main()
