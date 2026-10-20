"""Build fresh source-aligned WBC data using original BCCD boxes and CSV labels.

Never uses the supplied pretrained blood-cell classifiers or augmented derivatives.
All derivatives of one BCCD field inherit its existing detector split.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']


def main():
    output = ROOT/'data/WBC_verified'
    if output.exists():
        raise SystemExit('Output exists; use its saved manifest or choose a fresh output directory.')
    labels_path = ROOT/'data/BloodCellImages/dataset-master/dataset-master/labels.csv'
    with labels_path.open(newline='') as f:
        labels = {int(r['Image']): r['Category'].strip() for r in csv.DictReader(f)}
    rows, excluded = [], []
    for view in ('crop', 'field'):
        for split in ('train', 'val', 'test'):
            for cls in CLASSES:
                (output/view/split/cls).mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ('train','val','test'):
        for p in sorted((ROOT/'data/BCCD'/split/'img').glob('*.jpeg')):
            source_id = p.stem
            category = labels.get(int(source_id.split('_')[-1]), '')
            ann = json.loads((p.parent.parent/'ann'/(p.name+'.json')).read_text())
            boxes = [o['points']['exterior'] for o in ann['objects'] if o['classTitle']=='WBC']
            if category not in CLASSES or len(boxes) != 1:
                excluded.append(dict(source_id=source_id, split=split, category=category,
                                     wbc_boxes=len(boxes), reason='ambiguous_or_unsupported_label_or_box'))
                continue
            im = cv2.imread(str(p)); h,w = im.shape[:2]
            pixel_sha = hashlib.sha256(im.tobytes()).hexdigest()
            if pixel_sha in hashes:
                raise ValueError(f'Duplicate source image: {source_id} and {hashes[pixel_sha]}')
            hashes[pixel_sha] = (source_id,split)
            (xa,ya),(xb,yb) = boxes[0]
            x1,y1,x2,y2 = max(0,min(xa,xb)),max(0,min(ya,yb)),min(w,max(xa,xb)),min(h,max(ya,yb))
            # Ten native pixels of context, fixed before evaluation; shared by inference.
            pad=10
            crop=im[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]
            if crop.size == 0:
                raise ValueError(f'Empty crop: {p}')
            for view, image in [('crop',crop),('field',im)]:
                target=output/view/split/category/(source_id+'.png')
                if not cv2.imwrite(str(target),image):
                    raise OSError(target)
            rows.append(dict(source_id=source_id, split=split, label=category,
                             source_path=p.relative_to(ROOT).as_posix(), pixel_sha256=pixel_sha,
                             x1=x1,y1=y1,x2=x2,y2=y2,padding=pad))
    with (output/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=dict(total=len(rows),by_split=dict(Counter(r['split'] for r in rows)),
                 by_split_class=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
                 excluded=excluded,patient_disjoint=False,
                 label_policy='Exactly one supported CSV subtype and exactly one annotated WBC per field.',
                 source_policy='Inherit existing BCCD train/val/test partition; no precomputed augmentation.')
    (output/'preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='excluded'},indent=2))


if __name__=='__main__':
    main()
