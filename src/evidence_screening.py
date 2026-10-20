"""Inference for the ALL evidence-training experiment; one original-image CNN pass."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,time
import cv2,joblib,numpy as np
from .all_views import FrozenFeatures,LABELS
from .all_decisions import decide
ROOT=Path(__file__).resolve().parents[1]
DISPLAY={'Early':'Early pre-B','Pre':'Pre-B','Pro':'Pro-B'}


@lru_cache(maxsize=2)
def _load(serialized):
    r=json.loads(serialized)
    if r['labels']!=LABELS:raise ValueError('Unexpected ALL label order.')
    for a in [r['backbone'],r['head']]:
        if hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()!=a['sha256']:
            raise ValueError('Screening artifact differs from the evaluated model.')
    for name,digest in r['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Image processing differs from the evaluated implementation.')
    return FrozenFeatures(ROOT/r['backbone']['path']),joblib.load(ROOT/r['head']['path'])


def screen_all(image_path):
    start=time.perf_counter();im=cv2.imread(str(image_path))
    if im is None or min(im.shape[:2])<32 or np.ptp(im)==0:
        raise ValueError('A readable microscopy image of at least 32 pixels per side with visible structure is required.')
    registry=json.loads((ROOT/'config/evidence_registry.json').read_text())
    extractor,head=_load(json.dumps(registry,sort_keys=True))
    p=head.predict_proba(extractor.extract([im]))
    # Reuse established broad-category/subtype bookkeeping with a single posterior.
    d=decide(p,p,registry['thresholds'])
    accepted=bool(d['binary_accepted'][0]);positive=bool(d['binary'][0]);sub=bool(d['subtype_accepted'][0])
    status='review' if not accepted else 'all_pattern' if positive else 'benign_pattern'
    label={'review':'Screening result requires review','all_pattern':'ALL-like image pattern flagged',
           'benign_pattern':'Benign-like image pattern'}[status]
    subtype=LABELS[int(d['subtype'][0])] if sub else None
    return dict(task='acute_lymphoblastic_leukemia_screening',method='area_matched_evidence_training',
                image_name=Path(image_path).name,status=status,label=label,
                all_model_support=float(1-p[0,0]),binary_accepted=accepted,
                subtype=subtype,subtype_display=DISPLAY.get(subtype),subtype_accepted=sub,
                subtype_status='reported' if sub else 'review' if accepted and positive else 'not_applicable',
                views={'Evidence-trained head':{name:float(p[0,i]) for i,name in enumerate(LABELS)}},
                elapsed_seconds=time.perf_counter()-start,model_provenance=registry,
                diagnosis={'status':'not_assessed'},
                scope='Exploratory ALL image screening. The evaluation partition was used in earlier development; patient independence and clinical performance are not established.')


def analyze_combined(image_path):
    from .pipeline import analyze,summarize_counts
    cells=analyze(image_path)
    for row in cells['detections']:
        if row['cell_type']=='WBC':
            row.pop('subtype',None);row['subtype_status']='not_assessed_in_combined_mode'
    cells['cell_counts']=summarize_counts(cells['detections'])
    return dict(task='blood_cells_and_all_screening',image_name=Path(image_path).name,
                cell_analysis=cells,leukemia_screening=screen_all(image_path),
                scope='Cell detection and whole-image ALL screening were evaluated separately; joint clinical accuracy is not established.')
