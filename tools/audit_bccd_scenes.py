"""Nominate related BCCD fields using perceptual hashes and affine registration.

Candidates are retained as evidence for visual review. Conservative exclusion
of held-out fields related to a training or validation field is permitted before
test prediction, without moving any test image into model fitting.
"""
from pathlib import Path
import csv
import json
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_near_duplicates import phash


def main():
    rows=[]
    sift=cv2.SIFT_create(nfeatures=1500)
    for split in ('train','val','test'):
        for p in sorted((ROOT/'data/BCCD_YOLO'/split/'images').glob('*.jpeg')):
            image=cv2.imread(str(p)); gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kp,desc=sift.detectAndCompute(gray,None)
            rows.append(dict(source_id=p.stem,split=split,path=p.relative_to(ROOT).as_posix(),
                             hashes=[phash(np.rot90(im,k)) for im in (image,cv2.flip(image,1)) for k in range(4)],
                             points=np.array([x.pt for x in kp],np.float32),descriptors=desc,shape=gray.shape))
    candidates=[]
    matcher=cv2.BFMatcher()
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            distance=min((h^b['hashes'][0]).bit_count() for h in a['hashes'])
            if distance>16:
                continue
            matches=matcher.knnMatch(a['descriptors'],b['descriptors'],k=2)
            good=[m for m,n in matches if m.distance<0.75*n.distance]
            if len(good)<12:
                continue
            pa=np.array([a['points'][m.queryIdx] for m in good])
            pb=np.array([b['points'][m.trainIdx] for m in good])
            matrix,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=3.0)
            if matrix is None:
                continue
            mask=inliers.ravel().astype(bool); n=int(mask.sum()); ratio=n/len(good)
            area_a=cv2.contourArea(cv2.convexHull(pa[mask]))/np.prod(a['shape']) if n>=3 else 0.
            area_b=cv2.contourArea(cv2.convexHull(pb[mask]))/np.prod(b['shape']) if n>=3 else 0.
            if n>=12 and ratio>=0.6 and min(area_a,area_b)>=0.1:
                candidates.append(dict(a={k:a[k] for k in ('source_id','split','path')},
                                       b={k:b[k] for k in ('source_id','split','path')},
                                       phash_distance=distance,inliers=n,inlier_fraction=ratio,
                                       covered_fraction_a=float(area_a),covered_fraction_b=float(area_b),
                                       affine=matrix.tolist()))
    parent={r['source_id']:r['source_id'] for r in rows}
    def find(x):
        while parent[x]!=x:
            x=parent[x]
        return x
    for pair in candidates:
        a=find(pair['a']['source_id']); b=find(pair['b']['source_id']); parent[b]=a
    groups={}
    for r in rows:
        groups.setdefault(find(r['source_id']),[]).append(r)
    exclusions=[]
    priority={'train':0,'val':1,'test':2}
    for group in groups.values():
        earliest=min(priority[r['split']] for r in group)
        for r in group:
            if priority[r['split']]>earliest:
                exclusions.append(dict(source_id=r['source_id'],split=r['split'],
                                       reason='related scene occurs in an earlier fitting or selection split',
                                       related_sources=[x['source_id'] for x in group if x['source_id']!=r['source_id']]))
    output=dict(criteria=dict(phash_max=16,ratio_test=0.75,ransac_pixels=3,min_inliers=12,
                              min_inlier_fraction=0.6,min_field_coverage=0.1),
                candidates=candidates,exclusions=exclusions,
                limitation='Content screening cannot establish patient-level independence; accepted candidates also need visual review.')
    (ROOT/'audit/bccd_scene_audit.json').write_text(json.dumps(output,indent=2))
    with (ROOT/'data/WBC_verified/manifest.csv').open() as f:
        manifest=list(csv.DictReader(f))
    excluded={r['source_id'] for r in exclusions}
    eligible=[r for r in manifest if r['source_id'] not in excluded]
    with (ROOT/'data/WBC_verified/evaluation_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(eligible)
    print('Pairs',len(candidates),'exclusions',exclusions,flush=True)


if __name__=='__main__':
    main()
