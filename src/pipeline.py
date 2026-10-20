"""Source-aligned research pipeline with one record per detected object."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import time
import cv2
import numpy as np

try:
    from .features import measure_box
except ImportError:
    from features import measure_box

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'audit'))
SUBTYPES=('EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL')
ALL_DISPLAY={'Benign':'Benign dataset category', 'Early':'Early pre-B dataset category',
             'Pre':'Pre-B dataset category','Pro':'Pro-B dataset category'}


def load_registry():
    path=ROOT/'config/model_registry.json'
    if not path.is_file():
        raise FileNotFoundError('Verified model registry missing. Run tools/evaluate_verified.py first.')
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def _load_model(path, expected_sha, task, names):
    from ultralytics import YOLO
    import ultralytics.utils.torch_utils as torch_utils
    torch_utils.NUM_THREADS=2
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Model checkpoint missing: {p}')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError(f'Model does not match evaluated checkpoint: {p}')
    model=YOLO(str(p))
    if model.task!=task or tuple(model.names.values())!=tuple(names):
        raise ValueError(f'Wrong model task or labels: {p}')
    return model


def get_model(key, registry):
    r=registry[key]
    return _load_model(str(ROOT/r['path']),r['sha256'],r['task'],tuple(r['names']))


def read_image(path):
    image=cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def padded_crop(image, box, pad=10):
    h,w=image.shape[:2]
    x1,y1,x2,y2=map(int,box)
    return image[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]


def classify_images(model, images):
    if not images:
        return []
    results=model.predict(images,imgsz=224,device='cpu',verbose=False)
    output=[]
    for r in results:
        probabilities=r.probs.data.cpu().numpy().astype(float)
        k=int(np.argmax(probabilities))
        output.append(dict(label=model.names[k],score=float(probabilities[k]),
                           probabilities={model.names[i]:float(p) for i,p in enumerate(probabilities)}))
    return output


def summarize_counts(records):
    counts={name:0 for name in ('RBC','WBC','Platelet')}
    subtype_counts={name:0 for name in SUBTYPES}
    review=0
    for row in records:
        counts[row['cell_type']]+=1
        if row['cell_type']=='WBC':
            if row.get('subtype_status')=='accepted':
                subtype_counts[row['subtype']['label']]+=1
            else:
                review+=1
    return dict(**counts,total=sum(counts.values()),wbc_subtypes=subtype_counts,wbc_review=review)


def analyze(image_path, log=None, model_provider=None, registry=None):
    start=time.perf_counter()
    emit=log or (lambda message:None)
    registry=registry or load_registry()
    provider=model_provider or get_model
    raw=read_image(image_path); h,w=raw.shape[:2]
    emit('Detecting cells in the original image.')
    detector=provider('detector',registry)
    result=detector.predict(raw,imgsz=640,conf=registry['detector']['confidence'],
                            iou=0.7,device='cpu',verbose=False)[0]
    records=[]; crops=[]; indices=[]
    for box in result.boxes:
        name=detector.names[int(box.cls[0])]
        if name not in ('RBC','WBC','Platelet'):
            raise ValueError(f'Unexpected detector label: {name}')
        x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
        xyxy=[max(0,x1),max(0,y1),min(w,x2),min(h,y2)]
        if xyxy[2]<=xyxy[0] or xyxy[3]<=xyxy[1]:
            continue
        row=dict(object_id=len(records),cell_type=name,box=xyxy,
                 detector_score=float(box.conf[0]),morphology=measure_box(raw,xyxy))
        if name=='WBC':
            crop=padded_crop(raw,xyxy)
            row['subtype_status']='review'
            if crop.size:
                crops.append(crop); indices.append(len(records))
        records.append(row)
    if crops:
        emit(f'Classifying {len(crops)} white blood cell crops.')
        classifier=provider('wbc',registry)
        predictions=classify_images(classifier,crops)
        threshold=registry['wbc']['review_threshold']
        for index,prediction in zip(indices,predictions):
            if prediction['label'] not in SUBTYPES:
                raise ValueError('Unexpected WBC classifier label.')
            records[index]['subtype']=prediction
            records[index]['subtype_status']='accepted' if prediction['score']>=threshold else 'review'
    return dict(schema_version=2,image_name=Path(image_path).name,image_size=[w,h],
                elapsed_seconds=time.perf_counter()-start,cell_counts=summarize_counts(records),
                detections=records,diagnosis={'status':'not_assessed'},
                model_provenance={k:registry[k] for k in ('detector','wbc')},
                scope='Image-field counts and research predictions; not a complete blood count or diagnosis.')


def classify_single(image_path, task='wbc'):
    if task not in ('wbc','all_exploratory'):
        raise ValueError(task)
    registry=load_registry(); model=get_model(task,registry)
    prediction=classify_images(model,[read_image(image_path)])[0]
    threshold=registry[task].get('review_threshold')
    return dict(task=task,prediction=prediction,
                display_label=ALL_DISPLAY[prediction['label']] if task=='all_exploratory' else prediction['label'],
                status='exploratory' if threshold is None else
                       ('accepted' if prediction['score']>=threshold else 'review'),
                diagnosis={'status':'not_assessed'},model_provenance=registry[task])


def annotate(image, report):
    result=image.copy()
    colors={'RBC':(70,130,220),'WBC':(220,130,30),'Platelet':(80,160,70)}
    for row in report['detections']:
        x1,y1,x2,y2=row['box']; name=row['cell_type']
        if name=='WBC':
            name=row.get('subtype',{}).get('label','WBC') if row.get('subtype_status')=='accepted' else 'WBC: review'
        cv2.rectangle(result,(x1,y1),(x2,y2),colors[row['cell_type']],2)
        cv2.putText(result,name,(x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.4,colors[row['cell_type']],1,cv2.LINE_AA)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--output',default='analysis.json')
    a=p.parse_args(); report=analyze(a.image,log=print)
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'Wrote {a.output}')
