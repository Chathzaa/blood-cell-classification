"""Evaluate frozen checkpoints; save every prediction behind the paper tables.

The primary run is seed 42, chosen before test evaluation. Additional seeds are
sensitivity experiments, never candidates selected by test performance.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'audit'))
CLASSES=['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
OUT=ROOT/'outputs/verified'


def json_write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def evaluation_context(manifest,detector_path):
    """Bind reusable predictions to their actual sources and operating settings."""
    files={detector_path,ROOT/'data/WBC_verified/evaluation_manifest.csv',
           Path(__file__).resolve(),ROOT/'src/pipeline.py'}
    files.update(p for p in (ROOT/'data/BCCD_YOLO/test').rglob('*')
                 if p.suffix.lower() in ('.jpg','.jpeg','.png','.txt'))
    for row in manifest:
        if row['split']=='train':
            continue
        files.add(ROOT/row['source_path'])
        for view in ('crop','field'):
            files.add(ROOT/'data/WBC_verified'/view/row['split']/row['label']/(row['source_id']+'.png'))
    import ultralytics
    import torch
    return dict(schema_version=1,ultralytics=ultralytics.__version__,torch=torch.__version__,
                detector_settings=dict(imgsz=640,confidence=0.25,nms_iou=0.7),
                classifier_settings=dict(imgsz=224,padding=10,match_iou=0.5),
                files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(files)})


def wilson(correct,n):
    if not n:
        return [None,None]
    z=1.959963984540054; p=correct/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-d),float(mid+d)]


def metrics(y,p,labels=CLASSES):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,matthews_corrcoef,confusion_matrix
    pr,re,f1,support=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    correct=int(np.sum(np.array(y)==np.array(p)))
    return dict(n=len(y),correct=correct,accuracy=float(accuracy_score(y,p)),
                accuracy_wilson95=wilson(correct,len(y)),macro_f1=float(np.mean(f1)),
                mcc=float(matthews_corrcoef(y,p)),labels=labels,
                confusion_matrix=confusion_matrix(y,p,labels=labels).tolist(),
                per_class={c:dict(precision=float(pr[i]),recall=float(re[i]),f1=float(f1[i]),n=int(support[i]))
                           for i,c in enumerate(labels)})


def predict(model,images,batch=32):
    from src.pipeline import classify_images
    predictions=[]
    for start in range(0,len(images),batch):
        predictions+=classify_images(model,images[start:start+batch])
    return predictions


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.


def choose_review_threshold(rows,target=0.90,minimum=20):
    # Validation-only policy; empirical correctness target is not a probability guarantee.
    for threshold in np.arange(0.50,1.00,0.01):
        accepted=[r for r in rows if r['score'] is not None and r['score']>=threshold]
        if len(accepted)>=minimum and np.mean([r['true']==r['predicted'] for r in accepted])>=target:
            return round(float(threshold),2)
    return 1.000001


def selective(rows,threshold):
    accepted=[r for r in rows if r['score'] is not None and r['score']>=threshold]
    correct=sum(r['true']==r['predicted'] for r in accepted)
    return dict(threshold=threshold,n=len(rows),accepted=len(accepted),correct=correct,
                coverage=len(accepted)/len(rows),accuracy=correct/len(accepted) if accepted else None,
                accuracy_wilson95=wilson(correct,len(accepted)))


def detect_fields(model,manifest):
    result={}
    for row in manifest:
        image=cv2.imread(str(ROOT/row['source_path']))
        start=time.perf_counter()
        r=model.predict(image,imgsz=640,conf=0.25,iou=0.7,verbose=False,device='cpu')[0]
        detections=[dict(box=b.xyxy[0].cpu().numpy().astype(float).tolist(),
                         score=float(b.conf[0]),label=model.names[int(b.cls[0])]) for b in r.boxes]
        result[row['source_id']]=dict(detections=detections,seconds=time.perf_counter()-start)
    return result


def evaluate_classifier(model,view,seed,manifest,detected):
    from src.pipeline import padded_crop
    outputs={}
    for split in ('val','test'):
        subset=[r for r in manifest if r['split']==split]
        native=[str(ROOT/'data/WBC_verified'/view/split/r['label']/(r['source_id']+'.png')) for r in subset]
        native_predictions=predict(model,native)
        native_rows=[dict(source_id=r['source_id'],split=split,true=r['label'],predicted=p['label'],
                          score=p['score'],probabilities=p['probabilities']) for r,p in zip(subset,native_predictions)]
        crop_images=[]; crop_info=[]; missed=[]
        for r in subset:
            gt=[int(r[k]) for k in ('x1','y1','x2','y2')]
            candidates=[d for d in detected[r['source_id']]['detections'] if d['label']=='WBC']
            overlaps=[iou(gt,d['box']) for d in candidates]
            best=int(np.argmax(overlaps)) if overlaps and max(overlaps)>=0.50 else None
            if best is None:
                missed.append(dict(source_id=r['source_id'],split=split,true=r['label'],
                                   predicted='MISSED',score=None,match_iou=None,kind='ground_truth'))
            raw=cv2.imread(str(ROOT/r['source_path']))
            for i,d in enumerate(candidates):
                crop_images.append(padded_crop(raw,d['box']))
                crop_info.append(dict(source_id=r['source_id'],split=split,
                                      true=r['label'] if i==best else 'UNMATCHED',
                                      box=d['box'],match_iou=overlaps[i],
                                      kind='ground_truth' if i==best else 'false_positive'))
        crop_predictions=predict(model,crop_images)
        cascade_rows=[dict(r,predicted=p['label'],score=p['score'],probabilities=p['probabilities'])
                      for r,p in zip(crop_info,crop_predictions)]+missed
        gt_rows=[r for r in cascade_rows if r['kind']=='ground_truth']
        cm=np.zeros((4,5),dtype=int)
        for r in gt_rows:
            cm[CLASSES.index(r['true']),CLASSES.index(r['predicted']) if r['predicted'] in CLASSES else 4]+=1
        per_class={}
        for c in CLASSES:
            tp=sum(r['true']==c and r['predicted']==c for r in cascade_rows)
            fp=sum(r['true']!=c and r['predicted']==c for r in cascade_rows)
            fn=sum(r['true']==c and r['predicted']!=c for r in cascade_rows)
            per_class[c]=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
                              recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        correct=sum(r['true']==r['predicted'] for r in gt_rows)
        outputs[split]=dict(native=metrics([r['true'] for r in native_rows],[r['predicted'] for r in native_rows]),
                            cascade=dict(n=len(gt_rows),correct=correct,accuracy=correct/len(gt_rows),
                                         accuracy_wilson95=wilson(correct,len(gt_rows)),
                                         macro_f1=float(np.mean([v['f1'] for v in per_class.values()])),
                                         missed=len(missed),false_positive_wbc=sum(r['kind']=='false_positive' for r in cascade_rows),
                                         confusion_matrix=cm.tolist(),per_class=per_class),
                            native_predictions=native_rows,cascade_predictions=cascade_rows)
    threshold=choose_review_threshold([r for r in outputs['val']['cascade_predictions'] if r['kind']=='ground_truth'])
    for split in ('val','test'):
        rows=[r for r in outputs[split]['cascade_predictions'] if r['kind']=='ground_truth']
        outputs[split]['selective']=selective(rows,threshold)
        outputs[split]['selective']['accepted_false_positive_wbc']=sum(
            r['kind']=='false_positive' and r['score']>=threshold for r in outputs[split]['cascade_predictions'])
        outputs[split]['threshold_sensitivity']=[selective(rows,t) for t in [0.,0.5,0.7,0.8,0.9,0.95,0.99]]
    return dict(view=view,seed=seed,review_threshold=threshold,**outputs)


def detection_metrics(model):
    import yaml
    config=dict(path=(ROOT/'data/BCCD_YOLO').as_posix(),train='train/images',val='val/images',test='test/images',
                names={0:'RBC',1:'WBC',2:'Platelet'})
    config_path=OUT/'bccd_evaluation.yaml'; config_path.write_text(yaml.safe_dump(config))
    m=model.val(data=str(config_path),split='test',imgsz=640,batch=8,device='cpu',workers=0,
                conf=0.001,iou=0.7,plots=True,project=str(OUT),name='detector_test',exist_ok=True,verbose=False)
    values=dict(metrics={k:float(v) for k,v in m.results_dict.items()},
                per_class=[dict(name=model.names[int(k)],precision=float(m.box.p[i]),recall=float(m.box.r[i]),
                                ap50=float(m.box.ap50[i]),ap50_95=float(m.box.ap[i]))
                           for i,k in enumerate(m.box.ap_class_index)],
                precision_recall_policy='Ultralytics confidence maximizing mean F1; separate from fixed 0.25 app threshold.')
    return values


def legacy_audit(YOLO):
    result={}
    for task,checkpoint,data,labels in [
        ('wbc','runs/classify/runs/stage2_cls_groupsplit/weights/best.pt','BloodCell_CLS_v2',CLASSES),
        ('all','runs/classify/runs/stage3_cls_dedup/weights/best.pt','Leukemia_CLS_v2',['Benign','Early','Pre','Pro'])]:
        model=YOLO(str(ROOT/checkpoint)); paths=sorted((ROOT/'data'/data/'val').rglob('*'))
        paths=[p for p in paths if p.suffix.lower() in ('.jpg','.jpeg','.png')]
        predictions=predict(model,[str(p) for p in paths])
        rows=[dict(path=p.relative_to(ROOT).as_posix(),true=p.parent.name,**r) for p,r in zip(paths,predictions)]
        result[task]=dict(checkpoint=checkpoint,sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
                          split_role='checkpoint_selection_validation_not_independent_test',
                          metrics=metrics([r['true'] for r in rows],[r['label'] for r in rows],labels),predictions=rows)
        print('Legacy',task,result[task]['metrics']['accuracy'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--legacy-only',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Recompute predictions and detector AP, replacing cached results.')
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); args=parser.parse_args()
    import torch
    from ultralytics import YOLO
    import ultralytics.utils.torch_utils as torch_utils
    torch_utils.NUM_THREADS=2
    torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    if args.legacy_only:
        json_write(OUT/'legacy_validation.json',legacy_audit(YOLO)); return
    with (ROOT/'data/WBC_verified/evaluation_manifest.csv').open() as f:
        manifest=list(csv.DictReader(f))
    detector_path=ROOT/'runs/detect/runs/stage1_detection/weights/best.pt'
    context=evaluation_context(manifest,detector_path)
    context_sha=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    context_path=OUT/'evaluation_context.json'
    if not args.refresh:
        has_cache=any((OUT/name).exists() for name in ('field_detections.json','detection_metrics.json'))
        if has_cache and (not context_path.exists() or json.loads(context_path.read_text())!=context):
            raise SystemExit('Cached evaluation has missing or changed provenance. Run again with --refresh to recompute it.')
    # Commit the provenance only after every requested evaluation succeeds.
    detector=YOLO(str(detector_path))
    detected_path=OUT/'field_detections.json'
    if detected_path.exists() and not args.refresh:
        detected=json.loads(detected_path.read_text())
    else:
        detected=detect_fields(detector,[r for r in manifest if r['split']!='train'])
        json_write(detected_path,detected)
    detection_path=OUT/'detection_metrics.json'
    if args.refresh or not detection_path.exists():
        json_write(detection_path,detection_metrics(detector))
    experiments=[]
    for seed in args.seeds:
        for view in ('crop','field'):
            checkpoint=ROOT/f'runs/verified/wbc_{view}_seed{seed}/weights/last.pt'
            completed=checkpoint.parent.parent/'experiment.json'
            if not completed.exists():
                raise FileNotFoundError(f'Experiment has not completed: {completed}')
            output=OUT/f'wbc_{view}_seed{seed}.json'
            if output.exists() and not args.refresh:
                experiment=json.loads(output.read_text())
                if (experiment['sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    or experiment.get('evaluation_context_sha256')!=context_sha):
                    raise SystemExit(f'Stale prediction cache: {output}. Recompute with --refresh.')
            else:
                model=YOLO(str(checkpoint))
                experiment=evaluate_classifier(model,view,seed,manifest,detected)
                experiment.update(checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                                  sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  evaluation_context_sha256=context_sha)
                json_write(output,experiment)
            experiments.append(experiment)
            print(view,seed,'native',experiment['test']['native']['accuracy'],
                  'cascade',experiment['test']['cascade']['accuracy'],flush=True)
    primary=next(e for e in experiments if e['seed']==42 and e['view']=='crop')
    all_path=ROOT/'runs/classify/runs/stage3_cls_dedup/weights/best.pt'
    registry=dict(detector=dict(path=detector_path.relative_to(ROOT).as_posix(),task='detect',
                               names=['RBC','WBC','Platelet'],sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),confidence=0.25),
                  wbc=dict(path=primary['checkpoint'],task='classify',names=CLASSES,sha256=primary['sha256'],
                           review_threshold=primary['review_threshold'],
                           review_policy='Minimum validation threshold on 0.50:0.01:0.99 with at least 20 accepted matches and 90% empirical correctness. No guarantee on new data.'),
                  all_exploratory=dict(path=all_path.relative_to(ROOT).as_posix(),task='classify',names=['Benign','Early','Pre','Pro'],
                                       sha256=hashlib.sha256(all_path.read_bytes()).hexdigest(),review_threshold=None,
                                       limitation='Selected validation results only; patient mapping unavailable; not applied to general WBC crops.'))
    (ROOT/'config').mkdir(exist_ok=True); json_write(ROOT/'config/model_registry.json',registry)
    summary=[dict(view=e['view'],seed=e['seed'],native=e['test']['native'],cascade=e['test']['cascade'],
                  selective=e['test']['selective']) for e in experiments]
    json_write(OUT/'summary.json',summary)
    json_write(context_path,context)


if __name__=='__main__':
    main()
