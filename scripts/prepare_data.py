"""Prepare portable dataset YAMLs and SeaDronesSee ODv2 labels."""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict,Counter
from pathlib import Path
import yaml
from dataset_tools import audit_split,sha256

EVAL_URL='https://raw.githubusercontent.com/Ben93kie/SeaDronesSee/main/OD/evaluation%20of%20OD%20V2/required%20files/instances_val_iscrowd.json'
NAMES=['swimmer','boat','jetski','life_saving_appliances','buoy']

def safe_child(root,name):
    f=(root/name).resolve()
    if not f.is_relative_to(root.resolve()):raise ValueError(f'Image path escapes image directory: {name}')
    return f

def prepare_sea(source,out,mode,evaluation_file,download,class_layout):
    if download:
        import requests
        evaluation_file=out/'annotations/instances_val_iscrowd.json'
        evaluation_file.parent.mkdir(parents=True,exist_ok=True)
        response=requests.get(EVAL_URL,timeout=120);response.raise_for_status()
        evaluation_file.write_bytes(response.content)
    if evaluation_file is None or not evaluation_file.is_file():
        raise FileNotFoundError('Provide --evaluation-annotations instances_val_iscrowd.json or --download-evaluation')
    if not download:
        target=out/'annotations/instances_val_iscrowd.json';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(evaluation_file,target);evaluation_file=target
    summary={}
    for split in ['train','val']:
        annfile=source/'annotations'/f'instances_{split}.json'
        payload=json.loads(annfile.read_text(encoding='utf-8'))
        cats={int(x['id']) for x in payload['categories']}
        if not set(range(1,6)).issubset(cats):raise ValueError(f'Expected ODv2 category ids 1..5: {annfile}')
        annotations=defaultdict(list)
        for ann in payload['annotations']:
            if ann['category_id'] not in range(6):raise ValueError('Unexpected ODv2 category')
            annotations[ann['image_id']].append(ann)
        image_names=set();counts=Counter();ignored=0
        for im in payload['images']:
            name=im['file_name'];src=safe_child(source/'images'/split,name)
            dst=safe_child(out/'images'/split,name)
            if not src.is_file():raise FileNotFoundError(src)
            stem=str(Path(name).with_suffix(''))
            if stem in image_names:raise ValueError('Duplicate label stem')
            image_names.add(stem);dst.parent.mkdir(parents=True,exist_ok=True)
            if mode=='copy':shutil.copy2(src,dst)
            else:
                try:os.link(src,dst)
                except OSError as exc:raise OSError('Hardlink failed; use --image-mode copy or place output-root on the source filesystem') from exc
            w,h=im['width'],im['height'];lines=[]
            for ann in annotations[im['id']]:
                cid=int(ann['category_id'])
                if cid==0 or ann.get('iscrowd',0):ignored+=1;continue
                x,y,bw,bh=map(float,ann['bbox'])
                x1,y1=max(0,x),max(0,y);x2,y2=min(w,x+bw),min(h,y+bh)
                if x2<=x1 or y2<=y1:raise ValueError(f'Invalid foreground bbox: {ann}')
                label_id=cid if class_layout=='original' else cid-1
                lines.append(f'{label_id} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}')
                counts[cid]+=1
            label=out/'labels'/split/Path(name).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True)
            label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        summary[split]={'images':len(image_names),'foreground_instances':sum(counts.values()),'excluded_training_ignore_or_crowd':ignored,'instances_by_official_category':dict(counts),'source_annotations_sha256':sha256(annfile)}
        if split=='val':
            gt=json.loads(evaluation_file.read_text(encoding='utf-8'))
            def image_signature(im):
                return (Path(im['file_name']).stem,im['width'],im['height'])
            if {i['id']:image_signature(i) for i in gt['images']}!={i['id']:image_signature(i) for i in payload['images']}:
                raise ValueError('Official evaluation annotations do not match the source validation image IDs/filenames')
    names=['ignored']+NAMES if class_layout=='original' else NAMES
    config={'path':out.as_posix(),'train':'images/train','val':'images/val','names':dict(enumerate(names)),
            'class_layout':class_layout,
            'dataset_id':'seadronessee','evaluation_annotations':evaluation_file.resolve().as_posix()}
    summary['evaluation_annotations_sha256']=sha256(evaluation_file)
    summary['evaluation_annotations_source']=EVAL_URL
    return config,summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['complex_sea','seadronessee'],required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--image-mode',choices=['hardlink','copy'],default='hardlink')
    p.add_argument('--class-layout',choices=['original','foreground'],default='original',help='original preserves official ids 1..5 and the reserved slot 0; foreground maps targets to 0..4')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--evaluation-annotations',type=Path)
    g.add_argument('--download-evaluation',action='store_true')
    a=p.parse_args();source=a.source_root.resolve();out=a.output_root.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new or empty output-root')
    out.mkdir(parents=True,exist_ok=True)
    if a.dataset=='complex_sea':
        summary={k:{x:y for x,y in audit_split(source,k).items() if x!='files'} for k in ['train','val','test']}
        config={'path':source.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':{0:'person_overboard'},'dataset_id':'complex_sea'}
    else:config,summary=prepare_sea(source,out,a.image_mode,a.evaluation_annotations,a.download_evaluation,a.class_layout)
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Dataset configuration:',out/'data.yaml')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
