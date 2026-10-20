"""Identical image transformations and frozen feature extraction for fitting and inference."""
from pathlib import Path
import hashlib
import os
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'audit'))
LABELS=['Benign','Early','Pre','Pro']


def prepare_view(image,view='original',perturbation='none'):
    if image is None or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('A decoded three-channel microscopy image is required.')
    image=cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)
    if perturbation=='blur':image=cv2.GaussianBlur(image,(5,5),1.0)
    elif perturbation=='colour':image=np.clip(image.astype(float)*np.array([0.90,1.00,1.10]),0,255).astype(np.uint8)
    elif perturbation!='none':raise ValueError(perturbation)
    if view=='stain':
        sat=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1]
        _,mask=cv2.threshold(sat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=np.where(mask[:,:,None]>0,image,np.uint8(127)).astype(np.uint8)
    elif view!='original':raise ValueError(view)
    return image


class FrozenFeatures:
    def __init__(self,path=None):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2)
        self.torch=torch
        self.path=Path(path or ROOT/'yolov8n-cls.pt')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.model=YOLO(str(self.path)).model.eval()
        self.captured=None
        self.hook=self.model.model[-1].linear.register_forward_pre_hook(self._capture)

    def _capture(self,module,args):self.captured=args[0].detach()

    def extract(self,images,view='original',perturbation='none',batch=32):
        output=[]
        for start in range(0,len(images),batch):
            items=[]
            for x in images[start:start+batch]:
                im=cv2.imread(str(x)) if isinstance(x,(str,Path)) else x
                im=prepare_view(im,view,perturbation)
                items.append(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)))
            tensor=self.torch.from_numpy(np.stack(items)).float()/255.
            with self.torch.inference_mode():self.model(tensor)
            output.append(self.captured.cpu().numpy())
        return np.concatenate(output).astype(np.float32)
