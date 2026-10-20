"""Prepare image groups and a frozen train/validation/test split for ALL screening."""
from pathlib import Path
from functools import lru_cache
from collections import Counter,defaultdict
import csv
import hashlib
import json
import sys
import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_near_duplicates import phash
OUT=ROOT/'data/ALL_verified'
CLASSES=['Benign','Early','Pre','Pro']


@lru_cache(maxsize=512)
def descriptors(path,transform):
    im=cv2.imread(str(ROOT/path))
    if transform>=4: im=cv2.flip(im,1)
    im=np.ascontiguousarray(np.rot90(im,transform%4))
    points,desc=cv2.SIFT_create(nfeatures=800).detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
    return np.array([p.pt for p in points],dtype=np.float32),desc,im.shape[:2]


def registered(a,b,transform):
    pa,da,sa=descriptors(a,transform);pb,db,sb=descriptors(b,0)
    if da is None or db is None or len(db)<2:return None
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<0.75*p[1].distance]
    if len(good)<12:return None
    x=np.array([pa[m.queryIdx] for m in good]);y=np.array([pb[m.trainIdx] for m in good])
    matrix,inliers=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=3.)
    if matrix is None:return None
    mask=inliers.ravel().astype(bool);n=int(mask.sum())
    coverage=min(cv2.contourArea(cv2.convexHull(x[mask]))/np.prod(sa),
                 cv2.contourArea(cv2.convexHull(y[mask]))/np.prod(sb)) if n>=3 else 0.
    if n>=12 and n/len(good)>=0.60 and coverage>=0.10:
        return dict(inliers=n,inlier_fraction=n/len(good),coverage=float(coverage))
    return None


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(2026)
    if OUT.exists():raise SystemExit('Existing ALL study preserved. Use its manifest or a fresh study copy.')
    rows=[];pixel_index={};duplicates=[];hash_rows=[]
    for label in CLASSES:
        for p in sorted((ROOT/'data/Leukemia/Original'/label).glob('*.jpg')):
            im=cv2.imread(str(p))
            if im is None:raise ValueError(p)
            pixel=hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
            path=p.relative_to(ROOT).as_posix()
            if pixel in pixel_index:
                prior=rows[pixel_index[pixel]]
                if prior['label']!=label:raise ValueError('Conflicting exact duplicate labels.')
                duplicates.append(dict(removed=path,retained=prior['path']));continue
            pixel_index[pixel]=len(rows)
            rows.append(dict(source_id=p.stem,label=label,path=path,pixel_sha256=pixel,
                             file_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            hash_rows.append([phash(np.rot90(v,k)) for v in (im,cv2.flip(im,1)) for k in range(4)])
    hashes=np.array(hash_rows,dtype=np.uint64);parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    accepted=[];n_candidates=0
    for i in range(len(rows)-1):
        distance=np.bitwise_count(hashes[i][None,:]^hashes[i+1:,0,None])
        best=distance.min(axis=1)
        for offset in np.where(best<=10)[0]:
            j=i+1+int(offset);d=int(best[offset]);n_candidates+=1
            transform=int(np.argmin(distance[offset]))
            details={} if d<=4 else registered(rows[i]['path'],rows[j]['path'],transform)
            if details is None:continue
            parent[find(j)]=find(i)
            accepted.append(dict(a=rows[i]['path'],b=rows[j]['path'],distance=d,
                                 method='conservative_hash_group' if d<=4 else 'affine_registration',**details))
        if i%400==0:print('Screened',i,'of',len(rows),'accepted pairs',len(accepted),flush=True)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(i)
    conflicts={root for root,indices in groups.items() if len({rows[i]['label'] for i in indices})>1}
    eligible=[]
    for root,indices in groups.items():
        if root in conflicts:continue
        group=min(rows[i]['source_id'] for i in indices)
        for i in indices:eligible.append(dict(rows[i],group=group))
    eligible.sort(key=lambda r:r['path'])
    y=[r['label'] for r in eligible];g=[r['group'] for r in eligible]
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
    for fold,(_,indices) in enumerate(splitter.split(np.zeros(len(y)),y,g)):
        for i in indices:eligible[int(i)]['split']='test' if fold==0 else 'val' if fold==1 else 'train'
    OUT.mkdir(parents=True)
    with (OUT/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(eligible[0]));w.writeheader();w.writerows(eligible)
    group_splits=defaultdict(set)
    for r in eligible:group_splits[r['group']].add(r['split'])
    assert all(len(s)==1 for s in group_splits.values())
    summary=dict(original_images=len(rows)+len(duplicates),unique_pixel_images=len(rows),
                 duplicates=duplicates,similarity_candidates=n_candidates,accepted_pairs=accepted,
                 groups=len(group_splits),multi_image_groups=sum(len(v)>1 for v in groups.values()),
                 quarantined_images=[rows[i] for root in conflicts for i in groups[root]],
                 counts=dict(Counter(r['split'] for r in eligible)),
                 class_counts=dict(Counter(r['split']+'/'+r['label'] for r in eligible)),
                 group_counts={s:len({r['group'] for r in eligible if r['split']==s}) for s in ('train','val','test')},
                 patient_independence_established=False,seed=2026,
                 limitations='Visual groups are not patient identifiers. Similarity screening is not exhaustive.')
    (OUT/'preparation.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'audit/all_group_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('duplicates','accepted_pairs','quarantined_images')},indent=2))


if __name__=='__main__':main()
