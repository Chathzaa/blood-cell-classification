"""Area-matched stain/context interventions and a fitted linear screening head."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.special import softmax


def stain_proposal(image):
    image=cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)
    saturation=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1]
    _,raw=cv2.threshold(saturation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    raw=cv2.dilate(raw,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask=raw>0
    if mask.sum()>mask.size//2:
        order=np.argsort(-saturation.ravel().astype(float),kind='stable')
        mask=np.zeros(mask.size,dtype=bool);mask[order[:mask.size//2]]=True
        mask=mask.reshape(saturation.shape)
    return image,mask


def intervention_views(image):
    image,region=stain_proposal(image)
    n=int(region.sum());context=np.zeros(region.shape,dtype=bool)
    if n:
        distance=cv2.distanceTransform((~region).astype(np.uint8),cv2.DIST_L2,5)
        candidates=np.flatnonzero(~region.ravel())
        order=np.argsort(-distance.ravel()[candidates],kind='stable')
        context.ravel()[candidates[order[:n]]]=True
    def erased(mask):
        result=image.copy();result[mask]=127;return result
    return dict(original=image,context=erased(context),full_context=erased(~region),
                region=erased(region)),dict(valid=bool(n),area=n,region=region,context=context)


def perturb(image,condition):
    image=cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)
    if condition=='none':return image
    if condition=='blur':return cv2.GaussianBlur(image,(5,5),1.0)
    if condition=='colour':return np.clip(image.astype(float)*[.90,1.,1.10],0,255).astype(np.uint8)
    if condition=='context_colour':
        _,mask=stain_proposal(image)
        changed=np.clip(image.astype(float)*[.75,1.,1.25],0,255).astype(np.uint8)
        return np.where(mask[:,:,None],image,changed)
    raise ValueError(condition)


@dataclass
class EvidenceHead:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    temperature: float = 1.

    def decision_function(self,features):
        return ((features-self.mean)/self.scale)@self.weight.T+self.bias

    def predict_proba(self,features):
        return softmax(self.decision_function(features)/self.temperature,axis=1)

    def predict(self,features):
        return self.decision_function(features).argmax(axis=1)
