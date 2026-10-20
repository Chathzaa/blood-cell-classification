"""Optional detector retraining with related scenes excluded from validation.

The paper evaluates the supplied frozen detector. Running this script creates a
new experiment; its results do not replace the paper unless separately evaluated.
"""
from pathlib import Path
import argparse
import json
import os
import yaml

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'audit'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--weights',default='yolov8n.pt')
    args=parser.parse_args()
    audit_path=ROOT/'audit/bccd_scene_audit.json'
    if not audit_path.exists():
        raise SystemExit('First run tools/audit_bccd_scenes.py and review its candidate pairs.')
    excluded={r['source_id'] for r in json.loads(audit_path.read_text())['exclusions']}
    configuration=ROOT/'config/detector_source_split'
    configuration.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        paths=[p for p in sorted((ROOT/'data/BCCD_YOLO'/split/'images').glob('*.jpeg')) if p.stem not in excluded]
        (configuration/(split+'.txt')).write_text('\n'.join(p.as_posix() for p in paths)+'\n')
    data=configuration/'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=configuration.as_posix(),train='train.txt',val='val.txt',test='test.txt',
                                       names={0:'RBC',1:'WBC',2:'Platelet'})))
    name=f'detector_retrained_seed{args.seed}'
    if (ROOT/'runs/verified'/name).exists():
        raise SystemExit('This experiment exists; choose another seed or a fresh project copy.')
    from ultralytics import YOLO
    model=YOLO(args.weights)
    model.train(data=str(data),epochs=args.epochs,imgsz=640,batch=16,device=args.device,workers=0,
                optimizer='Adam',lr0=0.001,cos_lr=True,fliplr=0.5,flipud=0.2,degrees=15,hsv_v=0.2,
                seed=args.seed,deterministic=True,patience=50,project=str(ROOT/'runs/verified'),
                name=name,exist_ok=False)


if __name__=='__main__':
    main()
