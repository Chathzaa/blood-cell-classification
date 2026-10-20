"""Verify the current combined-scope release from source manifests and predictions."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_all_study import base_metrics,review_metrics
from src.all_decisions import choose_thresholds
from src.all_views import LABELS
from check_leukemia_leakage import main as check_groups
from tools.evaluate_verified import evaluation_context


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check_groups()
    study=read(ROOT/'outputs/all_verified/study.json');registry=read(ROOT/'config/leukemia_registry.json')
    for r in [registry['backbone'],*registry['heads'].values()]:assert sha(ROOT/r['path'])==r['sha256']
    assert sha(ROOT/'src/all_views.py')==registry['transformation_sha256']
    context=read(ROOT/'outputs/all_verified/feature_context.json')
    assert context['manifest']==sha(ROOT/'data/ALL_verified/manifest.csv')
    assert context['transformation']==sha(ROOT/'src/all_views.py')
    assert context['backbone']==sha(ROOT/'yolov8n-cls.pt')
    assert all(sha(ROOT/path)==digest for path,digest in context['source_files'].items())
    assert hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()==study['selection']['feature_context_sha256']
    val=study['evaluation']['val_none']['predictions']
    pv=np.array([r['original'] for r in val]);qv=np.array([r['stain'] for r in val]);yv=np.array([LABELS.index(r['true']) for r in val])
    assert choose_thresholds(pv,qv,yv)==registry['thresholds']==study['selection']['thresholds']
    with (ROOT/'data/ALL_verified/manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    for name,e in study['evaluation'].items():
        split=name.split('_')[0];rows=e['predictions'];y=np.array([LABELS.index(r['true']) for r in rows])
        expected={r['source_id']:r['label'] for r in manifest if r['split']==split}
        assert {r['source_id']:r['true'] for r in rows}==expected
        p=np.array([r['original'] for r in rows]);q=np.array([r['stain'] for r in rows])
        assert np.allclose(p.sum(axis=1),1) and np.allclose(q.sum(axis=1),1)
        for key,probs in [('original',p),('stain',q),('fusion',(p+q)/2)]:
            assert base_metrics(y,probs)==e[key],(name,key)
        assert review_metrics(y,p,q,registry['thresholds'])==e['review'],name
    wbc_registry=read(ROOT/'config/model_registry.json')
    for r in wbc_registry.values():assert sha(ROOT/r['path'])==r['sha256']
    with (ROOT/'data/WBC_verified/evaluation_manifest.csv').open() as f:wbc_manifest=list(csv.DictReader(f))
    assert evaluation_context(wbc_manifest,ROOT/wbc_registry['detector']['path'])==read(ROOT/'outputs/verified/evaluation_context.json')
    report=dict(status='passed',scope='blood cell detection and ALL screening',
                all_test_images=643,all_test_groups=629,
                source_and_checkpoint_hashes_match=True,all_metrics_recomputed_from_predictions=True,
                thresholds_reproduced_from_validation_only=True,
                note='Patient independence is not established. Manuscript verification is separate.')
    (ROOT/'audit/all_release_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
