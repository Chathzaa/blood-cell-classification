"""Controlled field-versus-crop experiment. Training sees train and val only."""
import os
from pathlib import Path
import argparse
import json
import hashlib
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'audit'))
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--view', choices=['crop','field'], required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=60)
    args=parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    name=f'wbc_{args.view}_seed{args.seed}'
    output=ROOT/'runs/verified'/name
    if output.exists():
        raise SystemExit(f'Refusing to overwrite an experiment: {output}')
    weights=ROOT/'yolov8n-cls.pt'
    model=YOLO(str(weights))
    start=time.perf_counter()
    model.train(data=str(ROOT/'data/WBC_verified'/args.view), epochs=args.epochs,
                imgsz=224,batch=16,device='cpu',workers=0,seed=args.seed,
                deterministic=True,optimizer='AdamW',lr0=0.001,lrf=0.01,
                weight_decay=0.0005,dropout=0.2,patience=args.epochs,
                auto_augment=None,erasing=0.0,fliplr=0.5,flipud=0.5,
                scale=0.0,degrees=0.0,translate=0.0,hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,
                project=str(ROOT/'runs/verified'),name=name,exist_ok=False,
                plots=True,verbose=False)
    record=dict(view=args.view,seed=args.seed,epochs=args.epochs,
                elapsed_seconds=time.perf_counter()-start,
                pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                training_source='ImageNet initialization; no supplied blood-cell checkpoint reused.',
                model_selection='Fixed terminal epoch; use last.pt. Validation does not select the study checkpoint.',
                selected_checkpoint='last.pt')
    (output/'experiment.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
