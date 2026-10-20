"""Fit the preregistered AMER candidate and all controls, preserving prior studies."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import cv2,joblib,numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score,log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.all_views import FrozenFeatures,LABELS
from src.evidence_regularization import EvidenceHead,intervention_views,perturb
from tools.run_all_study import base_metrics
OUT=ROOT/'outputs/all_novel';MODELS=ROOT/'runs/all_novel'
FAMILIES=('original','context_augmentation','full_context_augmentation','amer','unmatched_amer')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def fit_head(x,xc,xr,y,valid,l2,ranking):
    import torch
    torch.set_num_threads(2)
    mean=x.mean(axis=0,dtype=np.float64);scale=x.std(axis=0,dtype=np.float64);scale[scale<1e-8]=1.
    arrays=[torch.tensor((a-mean)/scale,dtype=torch.float64) for a in (x,xc,xr)]
    original,context,region=arrays
    yy=torch.tensor(y,dtype=torch.long);valid_t=torch.tensor(valid,dtype=torch.float64)
    counts=np.bincount(y,minlength=4);weights=torch.tensor(len(y)/(4*counts[y]),dtype=torch.float64)
    n,d=x.shape;calls=[0]
    def objective(flat):
        theta=torch.tensor(flat,dtype=torch.float64,requires_grad=True)
        w=theta[:4*d].reshape(4,d);b=theta[4*d:]
        logits=original@w.T+b
        loss=(torch.nn.functional.cross_entropy(logits,yy,reduction='none')*weights).mean()
        if xc is not x:
            lc=context@w.T+b
            aug=(torch.nn.functional.cross_entropy(lc,yy,reduction='none')*weights).mean()
            loss=(loss+aug)/2
        if ranking:
            lr=region@w.T+b
            def margin(z):
                other=z.clone();other[torch.arange(n),yy]=-torch.inf
                return z[torch.arange(n),yy]-torch.logsumexp(other,dim=1)
            comparison=torch.nn.functional.softplus(1.-margin(lc)+margin(lr))
            loss=loss+ranking*(comparison*weights*valid_t).sum()/torch.clamp((weights*valid_t).sum(),min=1.)
        loss=loss+.5*l2*(w*w).sum()
        loss.backward();calls[0]+=1
        return float(loss.detach()),theta.grad.detach().numpy()
    result=minimize(objective,np.zeros(4*d+4),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=2000,ftol=1e-10,gtol=1e-6,maxls=30))
    head=EvidenceHead(mean,scale,result.x[:4*d].reshape(4,d),result.x[4*d:])
    return head,dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                     evaluations=calls[0],objective=float(result.fun))


def feature_cache(extractor,rows,original,context_sha):
    cache=OUT/'development_features.npz'
    if cache.exists():
        z=np.load(cache);assert str(z['context_sha256'])==context_sha
        return {k:z[k] for k in ('original','context','full_context','region','valid','area')}
    idx=[i for i,r in enumerate(rows) if r['split']!='test'];features={'original':original[idx]}
    accum={k:[] for k in ('context','full_context','region')};areas=[];valid=[]
    for start in range(0,len(idx),128):
        batch=idx[start:start+128];views=[intervention_views(cv2.imread(str(ROOT/rows[i]['path']))) for i in batch]
        for k in accum:accum[k].append(extractor.extract([v[0][k] for v in views]))
        valid.extend(v[1]['valid'] for v in views);areas.extend(v[1]['area'] for v in views)
        print('Development intervention features',start+len(batch),'/',len(idx),flush=True)
    features.update({k:np.concatenate(v) for k,v in accum.items()})
    features.update(valid=np.array(valid),area=np.array(areas))
    np.savez_compressed(cache,**features,context_sha256=context_sha)
    return features


def train(extractor,rows,original,context_sha):
    if (MODELS/'selection.json').exists():return json.loads((MODELS/'selection.json').read_text())
    f=feature_cache(extractor,rows,original,context_sha)
    dev=[r for r in rows if r['split']!='test'];y=np.array([LABELS.index(r['label']) for r in dev])
    ti=np.array([i for i,r in enumerate(dev) if r['split']=='train']);vi=np.array([i for i,r in enumerate(dev) if r['split']=='val'])
    x=f['original'][ti];xr=f['region'][ti];selection={}
    for family in FAMILIES:
        path=MODELS/(family+'_selection.json')
        if path.exists():selection[family]=json.loads(path.read_text());continue
        context='full_context' if family in ('full_context_augmentation','unmatched_amer') else 'context'
        xc=x if family=='original' else f[context][ti]
        candidates=[];heads=[]
        for l2 in (.0001,.001,.01):
            for ranking in ((.1,1.) if family in ('amer','unmatched_amer') else (0.,)):
                start=time.perf_counter()
                head,optimization=fit_head(x,xc,xr,y[ti],f['valid'][ti],l2,ranking)
                p=head.predict_proba(f['original'][vi])
                row=dict(l2=l2,ranking_weight=ranking,macro_f1=float(f1_score(y[vi],p.argmax(axis=1),average='macro')),
                         nll=float(log_loss(y[vi],p,labels=range(4))),optimization=optimization)
                heads.append(head);candidates.append(row)
                save(MODELS/(family+'_candidates.json'),candidates)
                print(family,row,'seconds',round(time.perf_counter()-start,1),flush=True)
        eligible=[i for i,c in enumerate(candidates) if c['optimization']['success']]
        if not eligible:raise RuntimeError('No converged candidate: '+family)
        best=max(eligible,key=lambda i:(candidates[i]['macro_f1'],-candidates[i]['nll'],candidates[i]['l2']))
        head=heads[best];logits=head.decision_function(f['original'][vi])
        t=minimize_scalar(lambda lt:log_loss(y[vi],softmax(logits/np.exp(lt),axis=1),labels=range(4)),bounds=(-3.,3.),method='bounded')
        head.temperature=float(np.exp(t.x))
        joblib.dump(head,MODELS/(family+'.joblib'))
        p=head.predict_proba(f['original'][vi]);threshold=1.01
        for v in np.arange(.5,1.,.01):
            keep=np.maximum(p[:,0],1-p[:,0])>=v;pred=(1-p[:,0]>=.5);true=y[vi]>0
            if sum(keep&true)>=20 and sum(keep&~true)>=20 and np.mean(pred[keep]==true[keep])>=.99:
                threshold=float(round(v,2));break
        choice=dict(candidates=candidates,selected=best,temperature=head.temperature,binary_threshold=threshold,
                    validation=base_metrics(y[vi],p),model_sha256=sha(MODELS/(family+'.joblib')),
                    validation_predictions=[dict(source_id=dev[int(i)]['source_id'],probability=pr.tolist()) for i,pr in zip(vi,p)])
        save(path,choice);selection[family]=choice
    record=dict(context_sha256=context_sha,protocol_sha256=sha(ROOT/'audit/NOVEL_METHOD_PROTOCOL.md'),
                training_source_sha256=sha(Path(__file__)),families=selection,
                test_status='Previously used partition; exploratory extension, not independent confirmation.')
    save(MODELS/'selection.json',record)
    print('All family choices frozen before new test evaluation.',flush=True)
    return record


def evaluate(extractor,rows,original,selection,context_sha):
    idx=np.array([i for i,r in enumerate(rows) if r['split']=='test']);rr=[rows[int(i)] for i in idx]
    y=np.array([LABELS.index(r['label']) for r in rr]);heads={k:joblib.load(MODELS/(k+'.joblib')) for k in FAMILIES}
    results=dict(selection=selection,context_sha256=context_sha,evaluation={},predictions={},bootstrap={})
    for condition in ('none','blur','colour','context_colour'):
        cache=OUT/(condition+'_test_features.npz')
        if condition=='none':features=original[idx]
        elif cache.exists():
            z=np.load(cache);assert str(z['context'])==context_sha;features=z['features']
        else:
            chunks=[]
            for start in range(0,len(rr),128):
                chunks.append(extractor.extract([perturb(cv2.imread(str(ROOT/r['path'])),condition) for r in rr[start:start+128]]))
            features=np.concatenate(chunks);np.savez_compressed(cache,features=features,context=context_sha)
        probs={k:h.predict_proba(features) for k,h in heads.items()}
        results['evaluation'][condition]={k:base_metrics(y,p) for k,p in probs.items()}
        results['predictions'][condition]=[dict(source_id=r['source_id'],group=r['group'],true=r['label'],
                                               **{k:p[i].tolist() for k,p in probs.items()}) for i,r in enumerate(rr)]
        groups=sorted({r['group'] for r in rr});members={g:[i for i,r in enumerate(rr) if r['group']==g] for g in groups}
        rng=np.random.default_rng(2026);differences=[]
        for _ in range(2000):
            take=np.concatenate([members[groups[int(i)]] for i in rng.integers(0,len(groups),len(groups))])
            acc={k:np.mean(p[take].argmax(axis=1)==y[take]) for k,p in probs.items()}
            differences.append([acc['amer']-acc['original'],acc['amer']-acc['context_augmentation']])
        results['bootstrap'][condition]={k:np.quantile(np.array(differences)[:,i],[.025,.975]).tolist()
                                         for i,k in enumerate(('amer_minus_original','amer_minus_context_augmentation'))}
        print('Evaluation',condition,{k:round(v['accuracy'],4) for k,v in results['evaluation'][condition].items()},flush=True)
    clean=results['predictions']['none'];context=results['predictions']['context_colour']
    results['context_prediction_change']={k:float(np.mean([np.argmax(a[k])!=np.argmax(b[k]) for a,b in zip(clean,context)])) for k in FAMILIES}
    save(OUT/'study.json',results)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','evaluate','all'],default='train');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    rows=list(csv.DictReader((ROOT/'data/ALL_verified/manifest.csv').open()))
    previous=json.loads((ROOT/'outputs/all_verified/feature_context.json').read_text())
    assert previous['manifest']==sha(ROOT/'data/ALL_verified/manifest.csv')
    assert all(sha(ROOT/r['path'])==r['file_sha256'] for r in rows)
    extractor=FrozenFeatures()
    context=dict(manifest=previous['manifest'],backbone=extractor.sha256,
                 transformations=sha(ROOT/'src/evidence_regularization.py'),
                 previous_feature_context=sha(ROOT/'outputs/all_verified/feature_context.json'))
    context_sha=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    original=np.load(ROOT/'outputs/all_verified/original_features.npz')['features']
    assert original.shape==(len(rows),1280)
    save(OUT/'context.json',context)
    with threadpool_limits(limits=2):
        selection=train(extractor,rows,original,context_sha) if args.stage!='evaluate' else json.loads((MODELS/'selection.json').read_text())
        assert selection['context_sha256']==context_sha
        if args.stage in ('evaluate','all'):evaluate(extractor,rows,original,selection,context_sha)


if __name__=='__main__':main()
