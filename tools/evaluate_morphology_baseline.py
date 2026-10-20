"""Fit a classical comparator without fitting transforms to held-out images."""
from pathlib import Path
import csv
import json
import sys
import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features import measure_box,FEATURE_NAMES
from tools.evaluate_verified import metrics,CLASSES


def main():
    with (ROOT/'data/WBC_verified/evaluation_manifest.csv').open() as f:
        rows=list(csv.DictReader(f))
    partitions={}
    for split in ('train','val','test'):
        selected=[r for r in rows if r['split']==split]
        values=[]
        for r in selected:
            p=ROOT/'data/WBC_verified/crop'/split/r['label']/(r['source_id']+'.png')
            image=cv2.imread(str(p)); h,w=image.shape[:2]
            m=measure_box(image,[0,0,w,h])
            values.append([m.get('features',{}).get(k) for k in FEATURE_NAMES])
        partitions[split]=(np.array(values,dtype=float),[r['label'] for r in selected],selected)
    x,y,_=partitions['train']; xv,yv,_=partitions['val']
    trials=[]; models=[]
    for c in (0.01,0.1,1.,10.):
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                            LogisticRegression(C=c,class_weight='balanced',max_iter=5000,random_state=42))
        model.fit(x,y); p=model.predict(xv)
        trials.append(dict(C=c,validation_macro_f1=float(f1_score(yv,p,labels=CLASSES,average='macro',zero_division=0))))
        models.append(model)
    best=int(np.argmax([t['validation_macro_f1'] for t in trials])); model=models[best]
    result=dict(features=FEATURE_NAMES,tuning=trials,selected_C=trials[best]['C'],predictions={})
    for split,(x,y,rs) in partitions.items():
        p=model.predict(x)
        result[split]=metrics(y,p.tolist())
        result['predictions'][split]=[dict(source_id=r['source_id'],true=r['label'],predicted=pred) for r,pred in zip(rs,p)]
    majority=max(set(partitions['train'][1]),key=partitions['train'][1].count)
    yt=partitions['test'][1]
    result['majority_baseline']=metrics(yt,[majority]*len(yt))
    out=ROOT/'outputs/verified'; out.mkdir(exist_ok=True,parents=True)
    (out/'morphology_baseline.json').write_text(json.dumps(result,indent=2))
    joblib.dump(model,out/'morphology_baseline.joblib')
    print(json.dumps({k:result[k] for k in ('selected_C','test','majority_baseline')},indent=2))


if __name__=='__main__':
    main()
