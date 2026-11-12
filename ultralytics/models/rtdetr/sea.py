"""SeaDronesSee ODv2 evaluation using official crowd-aware annotations."""
import json
from copy import deepcopy
from pathlib import Path
from .coco import COCORTDETRValidator,evaluate_coco
from .val import RTDETRValidator

class SeaDronesSeeValidator(COCORTDETRValidator):
    def get_stats(self):
        stats=RTDETRValidator.get_stats(self)
        if int(self.nc) not in (5,6):raise ValueError('Expected five foreground classes, optionally with reserved slot 0')
        file=Path(self.data['evaluation_annotations'])
        source=json.loads(file.read_text(encoding='utf-8'))
        names={Path(x['file_name']).stem:x for x in source['images']}
        if len(names)!=len(source['images']):raise ValueError('Duplicate official validation image basenames')
        mapping={x['id']:names[Path(x['file_name']).stem]['id'] for x in self.coco_images}
        if len(mapping)!=len(source['images']) or set(mapping.values())!={x['id'] for x in source['images']}:
            raise ValueError('Evaluation must cover each official validation image exactly once')
        preds=deepcopy(self.coco_predictions)
        if int(self.nc)==6:
            for p in preds:p['category_id']-=1
            preds=[p for p in preds if p['category_id'] in range(1,6)]
        for p in preds:p['image_id']=mapping[p['image_id']]
        source['categories']=[c for c in source['categories'] if int(c['id']) in range(1,6)]
        source.setdefault('info',{});source.setdefault('licenses',[])
        self.coco_ground_truth=source;self.coco_predictions=preds
        self.coco_metrics=evaluate_coco(source,preds)
        stats['metrics/mAP50(B)']=self.coco_metrics['AP50']
        stats['metrics/mAP50-95(B)']=self.coco_metrics['AP']
        stats['fitness']=self.coco_metrics['AP']
        return stats
