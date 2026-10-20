"""Screen for visually similar cross-split images; candidates require review.

This is a sensitivity check, not proof of patient independence. A 64-bit DCT
hash distance of four bits is used only to nominate candidates, never to claim
ground-truth identity or to tune an evaluated model.
"""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def phash(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    values=cv2.dct(gray)[:8,:8].reshape(-1)
    bits=values>np.median(values[1:])
    bits[0]=False
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def main():
    summaries={}
    for dataset in ('BCCD_YOLO','Leukemia_CLS_v2'):
        base=ROOT/'data'/dataset
        paths=sorted(p for p in base.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        rows=[]
        for p in paths:
            image=cv2.imread(str(p))
            hashes=[]
            for reflected in (image,cv2.flip(image,1)):
                for rotation in range(4):
                    hashes.append(phash(np.rot90(reflected,rotation)))
            rows.append(dict(path=p.relative_to(ROOT).as_posix(),split=p.relative_to(base).parts[0],hashes=hashes))
        train=[r for r in rows if r['split']=='train']; held=[r for r in rows if r['split']!='train']
        candidates=[]
        for r in held:
            best=min(((min((a^t['hashes'][0]).bit_count() for a in r['hashes']),t) for t in train),key=lambda x:x[0])
            if best[0]<=4:
                candidates.append(dict(query=r['path'],reference=best[1]['path'],distance=best[0],
                                       interpretation='Visual similarity candidate, not confirmed duplicate or patient match.'))
        summaries[dataset]=dict(images=len(rows),held_out_images=len(held),candidates=candidates,
                                method='64-bit DCT hash; eight rotations/reflections; distance <=4')
        print(dataset,'candidates',len(candidates),flush=True)
    (ROOT/'audit/near_duplicate_candidates.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
