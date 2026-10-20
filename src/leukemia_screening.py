"""Validated study interface for image-level ALL screening and hierarchical review."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import time
import cv2
import joblib
import numpy as np
from .all_views import FrozenFeatures,LABELS
from .all_decisions import probabilities,decide

ROOT=Path(__file__).resolve().parents[1]
DISPLAY={'Early':'Early pre-B','Pre':'Pre-B','Pro':'Pro-B'}


@lru_cache(maxsize=2)
def _load(serialized_registry):
    registry=json.loads(serialized_registry)
    if registry['labels']!=LABELS:raise ValueError('Unexpected ALL category order.')
    for record in [registry['backbone'],*registry['heads'].values()]:
        path=ROOT/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'ALL artifact differs from the evaluated version: {path}')
    if hashlib.sha256((ROOT/'src/all_views.py').read_bytes()).hexdigest()!=registry['transformation_sha256']:
        raise ValueError('ALL image transformation differs from the evaluated version.')
    extractor=FrozenFeatures(ROOT/registry['backbone']['path'])
    heads={view:joblib.load(ROOT/r['path']) for view,r in registry['heads'].items()}
    for head in heads.values():
        if list(head.classes_)!=list(range(4)):raise ValueError('ALL head class order mismatch.')
    return extractor,heads


def screen_all(image_path):
    start=time.perf_counter();image=cv2.imread(str(image_path))
    if image is None or min(image.shape[:2])<32 or np.ptp(image)==0:
        raise ValueError('A readable microscopy image of at least 32 pixels per side with visible structure is required.')
    registry=json.loads((ROOT/'config/leukemia_registry.json').read_text())
    extractor,heads=_load(json.dumps(registry,sort_keys=True))
    pp={view:probabilities(head,extractor.extract([image],view),registry['heads'][view]['temperature'])
        for view,head in heads.items()}
    d=decide(pp['original'],pp['stain'],registry['thresholds'])
    accepted=bool(d['binary_accepted'][0]);positive=bool(d['binary'][0])
    subtype_accepted=bool(d['subtype_accepted'][0])
    if not accepted:status='review';label='Screening result requires review'
    elif positive:status='all_pattern';label='ALL-like image pattern flagged'
    else:status='benign_pattern';label='Benign-like image pattern'
    subtype=LABELS[int(d['subtype'][0])] if subtype_accepted else None
    return dict(task='acute_lymphoblastic_leukemia_screening',image_name=Path(image_path).name,
                status=status,label=label,all_model_support=float(d['all_score'][0]),
                binary_agreement=bool(d['binary_agreement'][0]),binary_accepted=accepted,
                subtype=subtype,subtype_display=DISPLAY.get(subtype),subtype_accepted=subtype_accepted,
                subtype_status='reported' if subtype_accepted else 'review' if accepted and positive else 'not_applicable',
                views={view:{name:float(prob[0,i]) for i,name in enumerate(LABELS)} for view,prob in pp.items()},
                elapsed_seconds=time.perf_counter()-start,model_provenance=registry,
                diagnosis={'status':'not_assessed'},
                scope='ALL source-domain image screening; other leukemia types and clinical diagnosis are not assessed.')


def analyze_combined(image_path):
    from .pipeline import analyze,summarize_counts
    cells=analyze(image_path)
    # Normal WBC subtype labels are not applied to potentially malignant cells.
    for row in cells['detections']:
        if row['cell_type']=='WBC':
            row.pop('subtype',None);row['subtype_status']='not_assessed_in_combined_mode'
    cells['cell_counts']=summarize_counts(cells['detections'])
    return dict(task='blood_cells_and_all_screening',image_name=Path(image_path).name,
                cell_analysis=cells,leukemia_screening=screen_all(image_path),
                scope='Components evaluated on their respective datasets; combined clinical accuracy is not established.')
