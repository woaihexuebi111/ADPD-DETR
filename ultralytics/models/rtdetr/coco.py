"""COCO bbox evaluation in original image coordinates, including AP selection."""

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .val import RTDETRValidator


def evaluate_coco(payload, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = payload
        ground_truth.createIndex()
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {**payload, "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(zip(("AP", "AP50", "AP75", "APS", "APM", "APL"),
                    (float(value) for value in evaluator.stats[:6])))


class COCORTDETRValidator(RTDETRValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.coco_images = []
        self.coco_annotations = []
        self.coco_predictions = []
        self.coco_metrics = {}

    def update_metrics(self, predictions, batch):
        for index, prediction in enumerate(predictions):
            image_id = len(self.coco_images) + 1
            height, width = batch["ori_shape"][index]
            self.coco_images.append({"id": image_id, "width": width, "height": height,
                                     "file_name": str(batch["im_file"][index])})
            selected = batch["batch_idx"] == index
            classes = batch["cls"][selected].view(-1).detach().cpu().tolist()
            for cls, box in zip(classes, batch["bboxes"][selected].detach().cpu().tolist()):
                xc, yc, bw, bh = box
                bw, bh = bw * width, bh * height
                self.coco_annotations.append({
                    "id": len(self.coco_annotations) + 1, "image_id": image_id,
                    "category_id": int(cls) + 1, "bbox": [xc * width - bw / 2, yc * height - bh / 2, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                })
            for x1, y1, x2, y2, score, cls in prediction.detach().float().cpu().tolist():
                x_scale, y_scale = width / self.args.imgsz, height / self.args.imgsz
                self.coco_predictions.append({
                    "image_id": image_id, "category_id": int(cls) + 1,
                    "bbox": [x1 * x_scale, y1 * y_scale, (x2 - x1) * x_scale, (y2 - y1) * y_scale],
                    "score": score,
                })
        super().update_metrics(predictions, batch)

    def get_stats(self):
        stats = super().get_stats()
        payload = {"info": {}, "licenses": [], "images": self.coco_images,
                   "annotations": self.coco_annotations,
                   "categories": [{"id": int(key) + 1, "name": name} for key, name in self.names.items()]}
        self.coco_ground_truth = payload
        self.coco_metrics = evaluate_coco(payload, self.coco_predictions)
        stats["metrics/mAP50(B)"] = self.coco_metrics["AP50"]
        stats["metrics/mAP50-95(B)"] = self.coco_metrics["AP"]
        stats["fitness"] = self.coco_metrics["AP"]
        return stats

    def eval_json(self, stats):
        # get_stats already performs the authoritative COCO evaluation.
        return stats
