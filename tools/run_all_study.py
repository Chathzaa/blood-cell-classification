"""Train new ALL heads, freeze validation choices, then evaluate the untouched test split."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import cv2
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.all_views import FrozenFeatures,LABELS
from src.all_decisions import probabilities,choose_thresholds,decide,decision_components
OUT=ROOT/'outputs/all_verified'
MODELS=ROOT/'runs/all_verified'


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def base_metrics(y,p):
    pred=p.argmax(axis=1);yb=y>0;pb=1-p[:,0];binary=pb>=0.5
    tn,fp,fn,tp=confusion_matrix(yb,binary,labels=[False,True]).ravel()
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                mcc=float(matthews_corrcoef(y,pred)),confusion=confusion_matrix(y,pred,labels=range(4)).tolist(),
                binary=dict(accuracy=float(np.mean(yb==binary)),sensitivity=float(tp/(tp+fn)),
                            specificity=float(tn/(tn+fp)),auroc=float(roc_auc_score(yb,pb)),
                            tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)),
                nll=float(log_loss(y,p,labels=range(4))))


def review_metrics(y,p,q,thresholds):
    d=decide(p,q,thresholds);yb=(y>0).astype(int);keep=d['binary_accepted'];sub=d['subtype_accepted']
    fusion=(p+q)/2
    confidence=np.maximum(1-fusion[:,0],fusion[:,0])
    # Matching coverage isolates ranking from the amount of rejected data.
    order=np.argsort(-confidence,kind='stable');matched=np.zeros(len(y),dtype=bool);matched[order[:int(keep.sum())]]=True
    def selected(mask,pred,true):
        n=int(mask.sum());correct=int(np.sum(pred[mask]==true[mask]))
        return dict(accepted=n,correct=correct,errors=n-correct,coverage=n/len(y),accuracy=correct/n if n else None)
    binary=selected(keep,d['binary'],yb)
    binary.update(class_coverage={LABELS[c]:float(np.mean(keep[y==c])) for c in range(4)},
                  retained_confusion=confusion_matrix(yb[keep],d['binary'][keep],labels=[0,1]).tolist(),
                  reviewed=int((~keep).sum()))
    raw_p=(1-p[:,0]>=0.5).astype(int)
    joint_score=d['binary_support']+d['binary_agreement'].astype(float)
    curves={}
    for name,scores,pred in [('original_confidence',np.maximum(p[:,0],1-p[:,0]),raw_p),
                             ('fusion_confidence',confidence,d['binary']),('view_consistency',joint_score,d['binary'])]:
        ranking=np.argsort(-scores,kind='stable');errors=(pred[ranking]!=yb[ranking]).astype(int)
        risk=np.cumsum(errors)/np.arange(1,len(y)+1)
        curves[name]=dict(coverage=(np.arange(1,len(y)+1)/len(y)).tolist(),risk=risk.tolist(),aurc=float(risk.mean()))
    return dict(binary=binary,subtype=selected(sub,d['subtype'],y),
                all_flag_subtype_review=int(np.sum(keep&(d['binary']==1)&~sub)),
                confidence_only_matched_coverage=selected(matched,d['binary'],yb),risk_coverage=curves)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refit',action='store_true',help='Explicitly repeat the frozen recipe and replace its model/result artifacts.')
    args=parser.parse_args()
    if (OUT/'study.json').exists() and not args.refit:
        raise SystemExit('Completed ALL study preserved. Verify its artifacts, or use --refit to deliberately recompute the fixed recipe.')
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    with (ROOT/'data/ALL_verified/manifest.csv').open() as f:rows=list(csv.DictReader(f))
    paths=[ROOT/r['path'] for r in rows];y=np.array([LABELS.index(r['label']) for r in rows])
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    context=dict(manifest=hashlib.sha256((ROOT/'data/ALL_verified/manifest.csv').read_bytes()).hexdigest(),
                 transformation=hashlib.sha256((ROOT/'src/all_views.py').read_bytes()).hexdigest(),
                 source_files={r['path']:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rows,paths)})
    assert all(context['source_files'][r['path']]==r['file_sha256'] for r in rows)
    extractor=FrozenFeatures();context['backbone']=extractor.sha256
    context_sha=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    heads={};temperatures={};all_features={};validation={}
    with threadpool_limits(limits=2):
        for view in ('original','stain'):
            cache=OUT/f'{view}_features.npz'
            if cache.exists():
                loaded=np.load(cache);assert str(loaded['context'])==context_sha;features=loaded['features']
            else:
                start=time.perf_counter();features=extractor.extract(paths,view)
                np.savez_compressed(cache,features=features,context=context_sha)
                print(view,'features',features.shape,'seconds',round(time.perf_counter()-start,1),flush=True)
            all_features[view]=features
            candidates=[];models=[]
            for c in (.01,.1,1.,10.):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2500,tol=1e-5,random_state=2026))
                model.fit(features[indices['train']],y[indices['train']])
                pred=model.predict(features[indices['val']]);score=float(f1_score(y[indices['val']],pred,average='macro'))
                candidates.append(dict(C=c,validation_macro_f1=score));models.append(model)
            best=max(range(len(candidates)),key=lambda i:(candidates[i]['validation_macro_f1'],-candidates[i]['C']))
            head=models[best];logits=head.decision_function(features[indices['val']])
            fitted=minimize_scalar(lambda lt:log_loss(y[indices['val']],softmax(logits/np.exp(lt),axis=1),labels=range(4)),bounds=(-3.,3.),method='bounded')
            temperature=float(np.exp(fitted.x));heads[view]=head;temperatures[view]=temperature
            joblib.dump(head,MODELS/f'{view}_head.joblib')
            validation[view]=dict(candidates=candidates,selected_C=candidates[best]['C'],temperature=temperature)
            print(view,'selected',validation[view],flush=True)
        pv=probabilities(heads['original'],all_features['original'][indices['val']],temperatures['original'])
        qv=probabilities(heads['stain'],all_features['stain'][indices['val']],temperatures['stain'])
        thresholds=choose_thresholds(pv,qv,y[indices['val']])
        selection=dict(validation=validation,thresholds=thresholds,feature_context_sha256=context_sha,
                       protocol='audit/LEUKEMIA_PROTOCOL.md',selection='Training and validation only; saved before test prediction.')
        save(MODELS/'selection.json',selection)
        print('Validation choices frozen:',thresholds,flush=True)
        results=dict(selection=selection,labels=LABELS,evaluation={})
        for split,perturbations in [('val',['none']),('test',['none','blur','colour'])]:
            idx=indices[split];ys=y[idx];records=[rows[int(i)] for i in idx]
            for perturbation in perturbations:
                pp={}
                for view in ('original','stain'):
                    x=all_features[view][idx] if perturbation=='none' else extractor.extract([paths[int(i)] for i in idx],view,perturbation)
                    pp[view]=probabilities(heads[view],x,temperatures[view])
                p,q=pp['original'],pp['stain'];fused=(p+q)/2
                key=split+'_'+perturbation
                output=dict(original=base_metrics(ys,p),stain=base_metrics(ys,q),fusion=base_metrics(ys,fused),
                            review=review_metrics(ys,p,q,thresholds),
                            predictions=[dict(source_id=r['source_id'],group=r['group'],true=r['label'],
                                              original=p[i].tolist(),stain=q[i].tolist()) for i,r in enumerate(records)])
                results['evaluation'][key]=output
                print(key,'original',output['original']['accuracy'],'fusion',output['fusion']['accuracy'],
                      'review',output['review']['binary'],flush=True)
        # Group bootstrap resamples source groups, retaining all members within each sampled group.
        test=results['evaluation']['test_none'];records=test['predictions'];ys=np.array([LABELS.index(r['true']) for r in records])
        p=np.array([r['original'] for r in records]);q=np.array([r['stain'] for r in records]);f=(p+q)/2
        groups=sorted({r['group'] for r in records});members={g:[i for i,r in enumerate(records) if r['group']==g] for g in groups}
        rng=np.random.default_rng(2026);boot=[]
        for _ in range(1000):
            sampled=np.concatenate([members[groups[int(i)]] for i in rng.integers(0,len(groups),len(groups))])
            a=ys[sampled];pred=f[sampled].argmax(axis=1);b=(1-f[sampled,0]>=.5);yb=a>0
            boot.append([np.mean(pred==a),np.mean(b[yb]),np.mean(~b[~yb])])
        test['group_bootstrap95']={name:np.quantile(np.array(boot)[:,i],[.025,.975]).tolist()
                                   for i,name in enumerate(('accuracy','sensitivity','specificity'))}
        save(OUT/'study.json',results)
        save(OUT/'feature_context.json',context)
        registry=dict(backbone=dict(path='yolov8n-cls.pt',sha256=extractor.sha256),
                      heads={v:dict(path=f'runs/all_verified/{v}_head.joblib',sha256=hashlib.sha256((MODELS/f'{v}_head.joblib').read_bytes()).hexdigest(),
                                    temperature=temperatures[v]) for v in heads},thresholds=thresholds,labels=LABELS,
                      transformation_sha256=context['transformation'],scope='Image-level ALL screening; patient independence and clinical performance unestablished.')
        save(ROOT/'config/leukemia_registry.json',registry)
    print('ALL study complete.',flush=True)


if __name__=='__main__':main()
