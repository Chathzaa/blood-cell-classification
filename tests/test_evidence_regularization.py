"""Protocol invariants for the region/context experiment, including degenerate inputs."""
import sys
from pathlib import Path
import cv2,joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evidence_regularization import intervention_views,perturb,EvidenceHead


def microscopy():
    image=np.full((224,224,3),225,np.uint8)
    cv2.circle(image,(100,100),30,(160,35,110),-1)
    return image


def test_interventions_have_equal_disjoint_areas_and_preserve_other_pixels():
    image=microscopy();v,m=intervention_views(image)
    assert m['area']>0 and m['area']==m['context'].sum()==m['region'].sum()
    assert not np.any(m['context']&m['region'])
    assert np.array_equal(v['context'][m['region']],image[m['region']])
    assert np.array_equal(v['region'][~m['region']],image[~m['region']])
    assert np.all(v['context'][m['context']]==127)


def test_half_image_cap_always_leaves_enough_matching_context():
    image=np.full((224,224,3),(180,30,120),np.uint8)
    v,m=intervention_views(image)
    assert 0<m['area']<=224*224//2 and m['area']==m['context'].sum()


def test_empty_stain_proposal_is_explicitly_unavailable():
    v,m=intervention_views(np.full((224,224,3),230,np.uint8))
    assert not m['valid'] and m['area']==0
    assert np.array_equal(v['context'],v['original'])


def test_context_stress_does_not_modify_proposed_evidence():
    image=microscopy();_,m=intervention_views(image)
    changed=perturb(image,'context_colour')
    assert np.array_equal(changed[m['region']],image[m['region']])
    assert np.any(changed[~m['region']]!=image[~m['region']])


def test_fitted_head_survives_serialization(tmp_path):
    head=EvidenceHead(np.zeros(3),np.ones(3),np.arange(12).reshape(4,3)*.1,np.zeros(4),1.4)
    x=np.array([[1.,-1.,2.],[0.,0.,0.]])
    path=tmp_path/'head.joblib';joblib.dump(head,path);loaded=joblib.load(path)
    assert np.allclose(head.predict_proba(x),loaded.predict_proba(x))
    assert np.allclose(loaded.predict_proba(x).sum(axis=1),1.)


def test_ranking_optimizer_fits_a_separable_control():
    from tools.run_evidence_study import fit_head
    x=np.tile(np.eye(4),(8,1));y=np.tile(np.arange(4),8)
    head,optimization=fit_head(x,x.copy(),np.zeros_like(x),y,np.ones(len(y),bool),.001,.1)
    assert optimization['success']
    assert np.array_equal(head.predict(x),y)


def test_current_inference_matches_recorded_test_probabilities():
    import csv,json
    from src.evidence_screening import screen_all
    study=json.loads((ROOT/'outputs/all_novel/study.json').read_text())
    expected=study['predictions']['none'][0]
    rows=list(csv.DictReader((ROOT/'data/ALL_verified/manifest.csv').open()))
    row=next(r for r in rows if r['source_id']==expected['source_id'])
    result=screen_all(ROOT/row['path'])
    assert np.allclose(list(result['views']['Evidence-trained head'].values()),expected['amer'],atol=1e-5)
    assert result['diagnosis']['status']=='not_assessed'


def test_current_inference_rejects_blank_images(tmp_path):
    import pytest
    from src.evidence_screening import screen_all
    path=tmp_path/'blank.png';cv2.imwrite(str(path),np.zeros((224,224,3),np.uint8))
    with pytest.raises(ValueError,match='visible structure'):screen_all(path)
