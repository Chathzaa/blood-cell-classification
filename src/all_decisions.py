"""View-consistent hierarchical screening decisions, separate from clinical diagnosis."""
import numpy as np
from scipy.special import softmax


def probabilities(head,features,temperature=1.):
    return softmax(head.decision_function(features)/temperature,axis=1)


def decision_components(p,q):
    p=np.asarray(p);q=np.asarray(q)
    bp=1-p[:,0];bq=1-q[:,0];predp=bp>=0.5;predq=bq>=0.5
    binary=(bp+bq)/2>=0.5
    supportp=np.where(binary,bp,1-bp);supportq=np.where(binary,bq,1-bq)
    pc=p[:,1:]/np.maximum(1e-12,1-p[:,0,None])
    qc=q[:,1:]/np.maximum(1e-12,1-q[:,0,None])
    subtype=np.argmax((pc+qc)/2,axis=1)+1
    k=np.arange(len(p))
    return dict(binary=binary.astype(int),all_score=(bp+bq)/2,
                binary_agreement=predp==predq,binary_support=np.minimum(supportp,supportq),
                subtype=subtype,subtype_agreement=np.argmax(pc,axis=1)==np.argmax(qc,axis=1),
                subtype_support=np.minimum(pc[k,subtype-1],qc[k,subtype-1]))


def choose_thresholds(p,q,y):
    d=decision_components(p,q);y=np.asarray(y);yb=(y>0).astype(int)
    binary_threshold=1.000001
    for threshold in np.arange(0.50,0.951,0.05):
        kept=d['binary_agreement']&(d['binary_support']>=threshold)
        if min(np.sum(kept&(yb==c)) for c in (0,1))>=20 and np.mean(d['binary'][kept]==yb[kept])>=0.99:
            binary_threshold=round(float(threshold),2);break
    binary_kept=d['binary_agreement']&(d['binary_support']>=binary_threshold)&(d['binary']==1)
    subtype_threshold=1.000001
    for threshold in np.arange(0.50,0.951,0.05):
        kept=binary_kept&d['subtype_agreement']&(d['subtype_support']>=threshold)
        if min(np.sum(kept&(d['subtype']==c)) for c in (1,2,3))>=20 and np.mean(d['subtype'][kept]==y[kept])>=0.95:
            subtype_threshold=round(float(threshold),2);break
    return dict(binary=binary_threshold,subtype=subtype_threshold)


def decide(p,q,thresholds):
    d=decision_components(p,q)
    d['binary_accepted']=d['binary_agreement']&(d['binary_support']>=thresholds['binary'])
    d['subtype_accepted']=d['binary_accepted']&(d['binary']==1)&d['subtype_agreement']&(d['subtype_support']>=thresholds['subtype'])
    return d
