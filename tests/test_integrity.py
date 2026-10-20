"""Regression tests for the faults found during the scientific audit."""
from pathlib import Path
import csv
import json
import sys
from types import SimpleNamespace
import numpy as np
import cv2
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features import stain_features,extract_all_features,FEATURE_NAMES,measure_box
from src.pipeline import summarize_counts,analyze,padded_crop
from tools.evaluate_verified import choose_review_threshold,iou


def test_count_conservation_with_review_and_nested_subtypes():
    rows=[{'cell_type':'RBC'},{'cell_type':'Platelet'},
          {'cell_type':'WBC','subtype_status':'accepted','subtype':{'label':'MONOCYTE'}},
          {'cell_type':'WBC','subtype_status':'review','subtype':{'label':'NEUTROPHIL'}}]
    counts=summarize_counts(rows)
    assert counts['total']==4
    assert sum(counts['wbc_subtypes'].values())+counts['wbc_review']==counts['WBC']==2


def test_stain_fraction_excludes_saturated_background():
    image=np.full((40,40,3),255,np.uint8)
    mask=np.zeros((40,40),np.uint8)
    cv2.circle(mask,(20,20),8,255,-1)
    image[mask==0]=[255,0,0]
    fraction,_=stain_features(image,mask)
    assert fraction==0
    image[20:23,20:23]=[150,20,100]
    fraction,_=stain_features(image,mask)
    assert 0<fraction<=1


def test_feature_dimensions_and_normalized_texture():
    image=np.full((64,64,3),240,np.uint8)
    cv2.circle(image,(32,32),18,(100,80,180),-1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cs,_=cv2.findContours((gray<200).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    values=extract_all_features(cs[0],image)
    assert len(values)==len(FEATURE_NAMES)==22
    assert np.isclose(sum(values[-4:]),1.0)
    assert 0<=values[5]<=1


def test_empty_region_never_fabricates_morphology():
    assert not measure_box(np.full((30,30,3),255,np.uint8),[0,0,30,30])['available']


def test_crop_clipping_and_invalid_input(tmp_path):
    image=np.zeros((20,30,3),np.uint8)
    assert padded_crop(image,[0,0,10,10]).shape==(20,20,3)
    with pytest.raises(ValueError,match='Cannot decode'):
        analyze(tmp_path/'missing.png',registry={'detector':{}})


def test_source_and_pixel_partitions_disjoint():
    with (ROOT/'data/WBC_verified/manifest.csv').open() as f:
        rows=list(csv.DictReader(f))
    for key in ('source_id','pixel_sha256'):
        splits={s:{r[key] for r in rows if r['split']==s} for s in ('train','val','test')}
        assert not splits['train']&splits['val']
        assert not splits['train']&splits['test']
        assert not splits['val']&splits['test']


def test_review_policy_refuses_unsupported_target():
    rows=[dict(true='A',predicted='B',score=0.99) for _ in range(25)]
    assert choose_review_threshold(rows)>1
    rows=[dict(true='A',predicted='A',score=0.81) for _ in range(25)]
    assert choose_review_threshold(rows)==0.5


def test_iou_requires_real_overlap():
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert iou([0,0,10,10],[11,11,20,20])==0


def test_pipeline_types_every_wbc_even_if_morphology_absent(tmp_path):
    import torch
    image=np.full((40,40,3),255,np.uint8)
    path=tmp_path/'blank.png'; cv2.imwrite(str(path),image)
    box=SimpleNamespace(cls=torch.tensor([1]),conf=torch.tensor([0.9]),
                        xyxy=torch.tensor([[5.,5.,30.,30.]]))
    class Detector:
        names={0:'RBC',1:'WBC',2:'Platelet'}
        def predict(self,*args,**kwargs):
            return [SimpleNamespace(boxes=[box])]
    class Classifier:
        names={0:'EOSINOPHIL',1:'LYMPHOCYTE',2:'MONOCYTE',3:'NEUTROPHIL'}
        def predict(self,*args,**kwargs):
            return [SimpleNamespace(probs=SimpleNamespace(data=torch.tensor([0.01,0.02,0.95,0.02])))]
    registry={'detector':{'confidence':0.25},'wbc':{'review_threshold':0.8}}
    result=analyze(path,registry=registry,model_provider=lambda key,_:Detector() if key=='detector' else Classifier())
    assert result['cell_counts']['WBC']==1
    assert result['cell_counts']['total']==1
    assert result['cell_counts']['wbc_subtypes']['MONOCYTE']==1
    assert result['diagnosis']['status']=='not_assessed'
    assert not result['detections'][0]['morphology']['available']
    json.dumps(result,allow_nan=False)


def test_detector_annotation_clips_corners_before_normalizing(tmp_path):
    from src.prepare_bccd import convert_supervisely_ann
    path=tmp_path/'annotation.json'
    path.write_text(json.dumps({'objects':[{'classTitle':'WBC','points':{'exterior':[[-5,2],[15,12]]}}]}))
    label=convert_supervisely_ann(path,20,20)[0]
    cls,x,y,w,h=map(float,label.split())
    assert cls==1
    assert np.allclose([x,y,w,h],[0.375,0.35,0.75,0.5])


def test_review_evaluation_excludes_registered_training_scene():
    with (ROOT/'data/WBC_verified/evaluation_manifest.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert 'BloodImage_00037' not in {r['source_id'] for r in rows}
    assert sum(r['split']=='val' for r in rows)==84
    assert sum(r['split']=='test' for r in rows)==63
