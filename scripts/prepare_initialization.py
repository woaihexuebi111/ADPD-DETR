"""Rebuild model initialization files from the public official RT-DETR R50 checkpoint."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics-config'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
URL='https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth'
EXPECTED_SHA256='dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    x=p.add_mutually_exclusive_group(required=True)
    x.add_argument('--official',type=Path)
    x.add_argument('--download',action='store_true')
    p.add_argument('--output-root',type=Path,default=ROOT/'initialization')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    official=a.official.expanduser().resolve() if a.official else out/'rtdetr_r50vd_6x_coco_from_paddle.pth'
    if a.download and not official.exists():
        import requests
        temp=official.with_suffix('.download')
        with requests.get(URL,stream=True,timeout=120) as response:
            response.raise_for_status()
            with temp.open('wb') as stream:
                for block in response.iter_content(1024*1024):stream.write(block)
        temp.rename(official)
    if not official.is_file():raise FileNotFoundError(official)
    from dataset_tools import sha256
    if sha256(official)!=EXPECTED_SHA256:raise ValueError('Official initialization checksum mismatch; use the documented RT-DETR R50 COCO checkpoint')
    baseline=out/'owrt_coco.pt';teacher=out/'teacher_coco.pt'
    if baseline.exists() or teacher.exists():raise FileExistsError('Choose a new output-root; initialization files already exist')
    from convert_official_owrt_r50 import convert
    from prepare_teacher_initialization import prepare_one
    convert(official,ROOT/'models/initialization/OWRT-R50-COCO.yaml',baseline,a.seed)
    import yaml
    teacher_config = yaml.safe_load((ROOT/'ultralytics/cfg/models/ADPD/ADPD-Teacher-R50.yaml').read_text(encoding='utf-8'))
    teacher_config['nc'] = 80
    config_path = out/'Teacher-COCO.yaml'
    config_path.write_text(yaml.safe_dump(teacher_config, sort_keys=False), encoding='utf-8')
    prepare_one(baseline,config_path,teacher,'ADPD teacher initialization',a.seed,False)
    info={'official_url':URL,'official_sha256':sha256(official),'seed':a.seed,
          'outputs':{f.name:sha256(f) for f in [baseline,teacher]}}
    (out/'initialization.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
