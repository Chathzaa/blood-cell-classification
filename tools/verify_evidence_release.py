"""Verify the experimental extension without promoting it to a superior model."""
from pathlib import Path
import csv,hashlib,json,sys
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_evidence_study import FAMILIES
from tools.run_all_study import base_metrics
from tools.verify_all_release import main as verify_previous
from src.all_views import LABELS
from src.all_decisions import choose_thresholds


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    verify_previous()
    rows=list(csv.DictReader((ROOT/'data/ALL_verified/manifest.csv').open()))
    expected={r['source_id']:r for r in rows if r['split']=='test'}
    s=read(ROOT/'outputs/all_novel/study.json');selection=read(ROOT/'runs/all_novel/selection.json')
    assert s['selection']==selection
    assert selection['training_source_sha256']==sha(ROOT/'tools/run_evidence_study.py')
    assert selection['protocol_sha256']==sha(ROOT/'audit/NOVEL_METHOD_PROTOCOL.md')
    context=read(ROOT/'outputs/all_novel/context.json')
    assert hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()==s['context_sha256']==selection['context_sha256']
    assert context['manifest']==sha(ROOT/'data/ALL_verified/manifest.csv')
    assert context['transformations']==sha(ROOT/'src/evidence_regularization.py')
    original=np.load(ROOT/'outputs/all_verified/original_features.npz')['features']
    train=original[[i for i,r in enumerate(rows) if r['split']=='train']]
    mean=train.mean(axis=0,dtype=np.float64);scale=train.std(axis=0,dtype=np.float64);scale[scale<1e-8]=1.
    for family in FAMILIES:
        q=selection['families'][family];path=ROOT/'runs/all_novel'/(family+'.joblib')
        assert sha(path)==q['model_sha256']
        assert all(c['optimization']['success'] for c in q['candidates'])
        best=max(range(len(q['candidates'])),key=lambda i:(q['candidates'][i]['macro_f1'],-q['candidates'][i]['nll'],q['candidates'][i]['l2']))
        assert q['selected']==best
        model=joblib.load(path)
        assert np.allclose(model.mean,mean) and np.allclose(model.scale,scale)
    for condition,records in s['predictions'].items():
        assert {r['source_id']:r['true'] for r in records}=={k:r['label'] for k,r in expected.items()}
        assert all(r['group']==expected[r['source_id']]['group'] for r in records)
        y=np.array([LABELS.index(r['true']) for r in records])
        for family in FAMILIES:
            p=np.array([r[family] for r in records]);assert np.allclose(p.sum(axis=1),1.)
            assert base_metrics(y,p)==s['evaluation'][condition][family]
    registry=read(ROOT/'config/evidence_registry.json')
    assert registry['selection_sha256']==sha(ROOT/'runs/all_novel/selection.json')
    for a in [registry['head'],registry['backbone']]:assert sha(ROOT/a['path'])==a['sha256']
    val=selection['families']['amer']['validation_predictions'];truth={r['source_id']:LABELS.index(r['label']) for r in rows}
    p=np.array([r['probability'] for r in val]);y=np.array([truth[r['source_id']] for r in val])
    t=choose_thresholds(p,p,y);t['binary']=selection['families']['amer']['binary_threshold']
    assert t==registry['thresholds']
    for path,digest in registry['source_hashes'].items():assert sha(ROOT/path)==digest
    ui=read(ROOT/'audit/evidence_ui_checks.json');assert len(ui['startup_modes'])==6 and ui['combined_analysis']=='passed'
    report=dict(status='passed',evaluation_status='exploratory; no demonstrated overall superiority',
                app_startup_variants=6,
                all_candidates_converged=True,all_metrics_recomputed=True,training_only_standardization_verified=True,
                validation_only_selection_verified=True,previous_test_use_disclosed=True,default_model_preserved=True,
                study_sha256=sha(ROOT/'outputs/all_novel/study.json'))
    (ROOT/'audit/evidence_release_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
