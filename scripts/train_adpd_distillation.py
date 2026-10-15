"""Train the RH-free ADPD student with the paper's DRD, TFD and PD objective."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-config"))

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.dgfe import PaperOWRTDMDGFE, PaperOWRTDMDGFERHFree
from ultralytics.nn.tasks import RTDETRDetectionModel, attempt_load_one_weight

DEFAULT_STUDENT = ROOT / "ultralytics/cfg/models/ADPD/ADPD-DETR-R50.yaml"
DEFAULT_DATA = ROOT / "configs/datasets/complex_sea.yaml"
CONDITION_Q = {"E01": .99, "E02": .96, "E03": .93, "E06": .95,
               "E10": .91, "E11": .94, "E12": .92, "E13": .93,
               "E14": .93, "E15": .93}
CONDITION_CODE_RE = re.compile(r"(?:^|[_-])(E\d{2})(?:[_\-.]|$)")


@dataclass
class DistillationConfig:
    teacher: Path
    drd: bool = True
    tfd: bool = True
    pd: bool = True
    lambda_d: float = .05
    lambda_b: float = .05
    lambda_tfd: float = .05
    lambda_cls: float = .05
    lambda_reg: float = .40
    alpha: float = .5
    beta: float = .2
    gamma: float = 1.
    weight_max: float = 2.
    topk: int = 100
    min_conf: float = .5
    student_layer: int = 8
    teacher_layer: int = 8
    gt_expand: float = 1.5
    gt_small_area: float = (32. / 640.) ** 2
    condition_q: bool = True
    default_q: float = .97


def per_image_minmax(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    flat = value.flatten(1)
    low = flat.amin(1).view(-1, 1, 1, 1)
    high = flat.amax(1).view(-1, 1, 1, 1)
    return (value - low) / (high - low + 1e-6)


def decoder_predictions(raw: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[1]
    if not isinstance(raw, tuple) or len(raw) != 5:
        raise RuntimeError("Unexpected RT-DETR decoder output")
    boxes, logits, _, _, dn = raw
    if dn is not None:
        count = int(dn["dn_num_split"][0])
        boxes, logits = boxes[:, :, count:], logits[:, :, count:]
    return (boxes[-1], logits[-1]) if boxes.ndim == 4 else (boxes, logits)


def prediction_distillation(student_raw: Any, teacher_raw: Any, cfg: DistillationConfig):
    s_boxes, s_logits = decoder_predictions(student_raw)
    t_boxes, t_logits = decoder_predictions(teacher_raw)
    if s_logits.shape != t_logits.shape or s_boxes.shape != t_boxes.shape:
        raise RuntimeError("PD requires matching decoder queries and detection classes")
    probabilities = t_logits.detach().sigmoid()
    confidence = probabilities.amax(-1)
    indices = confidence.topk(min(cfg.topk, confidence.shape[1]), dim=1).indices
    scores_idx = indices.unsqueeze(-1).expand(-1, -1, s_logits.shape[-1])
    boxes_idx = indices.unsqueeze(-1).expand(-1, -1, 4)
    conf = confidence.gather(1, indices)
    valid = (conf >= cfg.min_conf).to(s_logits.dtype)
    cls = F.binary_cross_entropy_with_logits(
        s_logits.gather(1, scores_idx), probabilities.gather(1, scores_idx), reduction="none",
    ).mean(-1)
    cls = (cls * valid).sum() / (valid.sum() + 1e-6)
    reg = F.smooth_l1_loss(
        s_boxes.gather(1, boxes_idx), t_boxes.detach().gather(1, boxes_idx), reduction="none",
    ).mean(-1)
    weights = conf * valid
    reg = (reg * weights).sum() / (weights.sum() + 1e-6)
    return cfg.lambda_cls * cls + cfg.lambda_reg * reg


def small_object_mask(batch: dict, size: tuple, ref: torch.Tensor, cfg: DistillationConfig):
    h, w = size
    mask = ref.new_zeros((batch["img"].shape[0], 1, h, w))
    for box, image_index in zip(batch["bboxes"].detach().cpu(), batch["batch_idx"].view(-1).cpu()):
        cx, cy, bw, bh = box.tolist()
        if bw <= 0 or bh <= 0 or bw * bh >= cfg.gt_small_area:
            continue
        bw, bh = bw * cfg.gt_expand, bh * cfg.gt_expand
        x1 = max(0, min(w - 1, int((cx - bw / 2) * w)))
        y1 = max(0, min(h - 1, int((cy - bh / 2) * h)))
        x2 = max(x1 + 1, min(w, int(torch.ceil(torch.tensor((cx + bw / 2) * w)))))
        y2 = max(y1 + 1, min(h, int(torch.ceil(torch.tensor((cy + bh / 2) * h)))))
        mask[int(image_index), :, y1:y2, x1:x2] = 1.
    return mask


_BASE_LOSS = RTDETRDetectionModel.loss
_TRAINERS: dict[int, "ADPDDistillationTrainer"] = {}
_PATCHED = False


def install_distillation_loss():
    global _PATCHED
    if _PATCHED:
        return

    def loss(model, batch, preds=None):
        trainer = _TRAINERS.get(id(model))
        if trainer is None or not model.training or preds is not None:
            return _BASE_LOSS(model, batch, preds=preds)
        trainer.features.clear()
        targets = {
            "cls": batch["cls"].to(batch["img"].device).long().view(-1),
            "bboxes": batch["bboxes"].to(batch["img"].device),
            "batch_idx": batch["batch_idx"].to(batch["img"].device).long().view(-1),
            "gt_groups": [(batch["batch_idx"] == i).sum().item() for i in range(len(batch["img"]))],
        }
        try:
            student_raw = model.predict(batch["img"], batch=targets)
            base, items = _BASE_LOSS(model, batch, preds=student_raw)
            with torch.no_grad():
                trainer.set_condition_quantiles(batch)
                teacher_raw = trainer.teacher_model.predict(batch["img"])
            cfg = trainer.distillation
            extra = base.new_zeros(())
            if cfg.drd:
                drd = trainer.response_loss()
                extra = extra + drd
                trainer.last_losses["DRD"] = float(drd.detach())
            if cfg.tfd:
                tfd = trainer.feature_loss(batch)
                extra = extra + tfd
                trainer.last_losses["TFD"] = float(tfd.detach())
            if cfg.pd:
                pd = prediction_distillation(student_raw, teacher_raw, cfg)
                extra = extra + pd
                trainer.last_losses["PD"] = float(pd.detach())
            return base + extra, items
        finally:
            trainer.teacher_branch.condition_quantiles = None
            trainer.features.clear()
            # The returned loss owns the graph; do not retain it in saved modules.
            trainer.student_branch.last_predicted_difference_map = None
            trainer.student_branch.last_predicted_filtration_map = None

    RTDETRDetectionModel.loss = loss
    _PATCHED = True


class ADPDDistillationTrainer(RTDETRTrainer):
    def __init__(self, *args, distillation: DistillationConfig, init_weights=None, **kwargs):
        self.distillation = distillation
        self.init_weights = init_weights
        self.teacher_model = None
        self.teacher_branch = None
        self.student_branch = None
        self.features = {}
        self.handles = []
        self.last_losses = {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if weights is None and self.init_weights is not None:
            initial, _ = attempt_load_one_weight(str(self.init_weights), device="cpu")
            model.load(initial, verbose=verbose)
        return model

    def _setup_train(self, world_size):
        if world_size > 1:
            raise ValueError("This distillation entry point supports one GPU, as configured in the paper")
        super()._setup_train(world_size)
        teacher, _ = attempt_load_one_weight(str(self.distillation.teacher), device=self.device)
        teacher.eval().requires_grad_(False)
        self.teacher_model = teacher
        student = self.model.module if hasattr(self.model, "module") else self.model
        self.student_branch = next((m for m in student.modules() if isinstance(m, PaperOWRTDMDGFERHFree)), None)
        self.teacher_branch = next((m for m in teacher.modules()
                                   if isinstance(m, PaperOWRTDMDGFE) and not isinstance(m, PaperOWRTDMDGFERHFree)), None)
        if self.student_branch is None or self.teacher_branch is None:
            raise ValueError("Expected a reconstruction teacher and an RH-free DDPM student")
        if student.model[-1].nc != teacher.model[-1].nc:
            raise ValueError("Teacher and student class counts must match the dataset")
        self.teacher_branch.threshold_mode = "adaptive_quantile"
        self.register_hooks()
        _TRAINERS[id(student)] = self
        install_distillation_loss()

    def register_hooks(self):
        self.remove_hooks()
        if not self.distillation.tfd:
            return
        student = self.model.module if hasattr(self.model, "module") else self.model
        for key, network, index in (("student", student, self.distillation.student_layer),
                                    ("teacher", self.teacher_model, self.distillation.teacher_layer)):
            if not 0 <= index < len(network.model):
                raise ValueError(f"Invalid {key} feature layer {index}")
            self.handles.append(network.model[index].register_forward_hook(
                lambda _module, _inputs, output, key=key: self.features.__setitem__(key, output)))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save_model(self):
        self.remove_hooks()
        try:
            return super().save_model()
        finally:
            self.register_hooks()

    def set_condition_quantiles(self, batch):
        cfg = self.distillation
        if not cfg.condition_q:
            self.teacher_branch.adaptive_quantile = cfg.default_q
            self.teacher_branch.condition_quantiles = None
            return
        paths = batch.get("im_file", [])
        if len(paths) != len(batch["img"]):
            raise ValueError("Condition-adaptive training requires image paths")
        quantiles = []
        for path in paths:
            match = CONDITION_CODE_RE.search(Path(path).name)
            quantiles.append(CONDITION_Q.get(match.group(1), cfg.default_q) if match else cfg.default_q)
        self.teacher_branch.condition_quantiles = quantiles

    def teacher_maps(self):
        difference = self.teacher_branch.last_difference_map
        binary = self.teacher_branch.last_binary_map
        if difference is None or binary is None:
            raise RuntimeError("Teacher difference/filtration maps were not captured")
        return per_image_minmax(difference), binary.detach().float()

    def response_loss(self):
        cfg = self.distillation
        d, b = self.teacher_maps()
        sd = self.student_branch.last_predicted_difference_map
        sb = self.student_branch.last_predicted_filtration_map
        if sd is None or sb is None or not sd.requires_grad or not sb.requires_grad:
            raise RuntimeError("DRD student maps must retain their autograd graph")
        d = F.interpolate(d, size=sd.shape[-2:], mode="bilinear", align_corners=False).to(sd)
        b = F.interpolate(b, size=sb.shape[-2:], mode="bilinear", align_corners=False).to(sb)
        return cfg.lambda_d * F.smooth_l1_loss(sd, d) + cfg.lambda_b * F.smooth_l1_loss(sb, b)

    def feature_loss(self, batch):
        cfg = self.distillation
        s, t = self.features["student"], self.features["teacher"].detach()
        if s.shape[1] != t.shape[1]:
            raise ValueError("TFD channel mismatch; use the projected student enhanced feature")
        size = s.shape[-2:]
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False).to(s)
        d, b = self.teacher_maps()
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False).to(s)
        b = F.interpolate(b, size=size, mode="bilinear", align_corners=False).to(s)
        mask = small_object_mask(batch, size, s, cfg)
        weights = (1. + cfg.alpha * d + cfg.beta * b + cfg.gamma * mask).clamp(1., cfg.weight_max).detach()
        differences = F.smooth_l1_loss(s, t, reduction="none").mean(1, keepdim=True)
        return cfg.lambda_tfd * (weights * differences).sum() / (weights.sum() + 1e-6)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="adpd_student")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--no-drd", action="store_true")
    parser.add_argument("--no-tfd", action="store_true")
    parser.add_argument("--no-pd", action="store_true")
    parser.add_argument("--lambda-dmap", type=float, default=.05)
    parser.add_argument("--lambda-filtration", type=float, default=.05)
    parser.add_argument("--lambda-tfd", type=float, default=.05)
    parser.add_argument("--lambda-cls", type=float, default=.05)
    parser.add_argument("--lambda-reg", type=float, default=.40)
    parser.add_argument("--kd-topk", type=int, default=100)
    parser.add_argument("--kd-min-conf", type=float, default=.5)
    parser.add_argument("--student-feature-layer", type=int, default=8)
    parser.add_argument("--teacher-feature-layer", type=int, default=8)
    parser.add_argument("--gt-expand", type=float, default=1.5)
    parser.add_argument("--no-condition-q", action="store_true")
    parser.add_argument("--default-condition-q", type=float, default=.97)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.student, args.teacher, args.data, args.init_weights):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.kd_topk < 1 or not 0 <= args.kd_min_conf <= 1 or not 0 < args.default_condition_q < 1:
        raise ValueError("Invalid query filtering or teacher quantile")
    cfg = DistillationConfig(
        teacher=args.teacher, drd=not args.no_drd, tfd=not args.no_tfd, pd=not args.no_pd,
        lambda_d=args.lambda_dmap, lambda_b=args.lambda_filtration,
        lambda_tfd=args.lambda_tfd, lambda_cls=args.lambda_cls, lambda_reg=args.lambda_reg,
        topk=args.kd_topk, min_conf=args.kd_min_conf, student_layer=args.student_feature_layer,
        teacher_layer=args.teacher_feature_layer, gt_expand=args.gt_expand,
        gt_small_area=(32. / args.imgsz) ** 2,
        condition_q=not args.no_condition_q, default_q=args.default_condition_q,
    )
    train_args = dict(model=str(args.student), data=str(args.data), epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
                      optimizer="AdamW", lr0=args.lr0, lrf=1., weight_decay=args.weight_decay,
                      seed=args.seed, amp=args.amp, pretrained=False, mosaic=0., mixup=0.,
                      close_mosaic=0, warmup_epochs=2000, patience=0, plots=False,
                      project=str(args.project), name=args.name, exist_ok=args.exist_ok,
                      save_period=args.save_period, val=True, split="val")
    trainer = ADPDDistillationTrainer(overrides=train_args, distillation=cfg, init_weights=args.init_weights)
    report = ROOT / "reports" / args.name
    report.mkdir(parents=True, exist_ok=True)
    metadata = {"started_at": datetime.now().isoformat(), "train_args": train_args,
                "distillation": asdict(cfg), "condition_q_map": CONDITION_Q,
                "init_weights": args.init_weights, "selection": "pycocotools bbox AP on validation split"}
    def record(filename):
        (report / filename).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    record("train_metadata_start.json")
    try:
        trainer.train()
        metadata.update(status="finished", best=str(trainer.best), last=str(trainer.last),
                        last_losses=trainer.last_losses)
        record("train_metadata_done.json")
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc))
        record("train_metadata_failed.json")
        raise
    finally:
        trainer.remove_hooks()
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        _TRAINERS.pop(id(model), None)


if __name__ == "__main__":
    main()
