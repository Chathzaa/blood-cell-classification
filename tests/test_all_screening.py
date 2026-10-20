"""Meaningful checks for grouped data and leukemia review semantics."""
from pathlib import Path
import csv
import json
import sys
import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.all_views import prepare_view
from src.all_decisions import decide,choose_thresholds


def test_all_source_groups_never_cross_partitions():
    with (ROOT/'data/ALL_verified/manifest.csv').open() as f:rows=list(csv.DictReader(f))
    for key in ('group','pixel_sha256'):
        sets={s:{r[key] for r in rows if r['split']==s} for s in ('train','val','test')}
        assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    assert len(rows)==3216 and sum(r['split']=='test' for r in rows)==643


def test_binary_disagreement_always_requires_review():
    p=np.array([[.99,.004,.003,.003]]);q=np.array([[.10,.8,.05,.05]])
    d=decide(p,q,{'binary':.5,'subtype':.5})
    assert not d['binary_accepted'][0] and not d['subtype_accepted'][0]


def test_subtype_disagreement_retains_all_screening_flag():
    p=np.array([[.01,.80,.10,.09]]);q=np.array([[.01,.10,.80,.09]])
    d=decide(p,q,{'binary':.5,'subtype':.5})
    assert d['binary_accepted'][0] and d['binary'][0]==1 and not d['subtype_accepted'][0]


def test_benign_result_never_receives_malignant_subtype():
    p=np.array([[.95,.02,.02,.01]])
    d=decide(p,p,{'binary':.5,'subtype':.5})
    assert d['binary_accepted'][0] and d['binary'][0]==0 and not d['subtype_accepted'][0]


def test_insufficient_validation_support_fails_closed():
    p=np.tile([.99,.004,.003,.003],(10,1));y=np.zeros(10,dtype=int)
    t=choose_thresholds(p,p,y)
    assert t['binary']>1 and t['subtype']>1


def test_stain_view_is_label_independent_and_retains_selected_pixels():
    image=np.full((224,224,3),220,np.uint8);cv2.circle(image,(112,112),35,(180,30,100),-1)
    stain=prepare_view(image,'stain')
    assert np.array_equal(stain[112,112],image[112,112])
    assert np.array_equal(stain[0,0],[127,127,127])
    assert np.array_equal(prepare_view(image,'original'),image)


def test_live_all_prediction_matches_saved_evidence():
    from src.leukemia_screening import screen_all
    study=json.loads((ROOT/'outputs/all_verified/study.json').read_text())
    expected=study['evaluation']['test_none']['predictions'][0]
    with (ROOT/'data/ALL_verified/manifest.csv').open() as f:
        row=next(r for r in csv.DictReader(f) if r['source_id']==expected['source_id'])
    result=screen_all(ROOT/row['path'])
    for view in ('original','stain'):
        assert np.allclose(list(result['views'][view].values()),expected[view],atol=1e-5)
    assert result['diagnosis']['status']=='not_assessed'


def test_empty_input_cannot_receive_leukemia_label(tmp_path):
    from src.leukemia_screening import screen_all
    p=tmp_path/'blank.png';cv2.imwrite(str(p),np.zeros((224,224,3),np.uint8))
    with pytest.raises(ValueError,match='visible structure'):screen_all(p)
