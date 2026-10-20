"""Descriptive image features. Stain proxies are not identified nuclei or diagnoses."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

FEATURE_NAMES = [
    'area', 'perimeter', 'circularity', 'eccentricity', 'solidity',
    'high_saturation_fraction', 'stain_boundary_irregularity',
    'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std',
    'glcm_energy', 'glcm_contrast', 'glcm_homogeneity', 'glcm_entropy', 'glcm_correlation',
    'lbp_bin0', 'lbp_bin1', 'lbp_bin2', 'lbp_bin3',
]


def geometric_features(contour):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    eccentricity = float('nan')
    if len(contour) >= 5:
        axes = cv2.fitEllipse(contour)[1]
        if max(axes) > 0:
            eccentricity = float(np.sqrt(max(0.0, 1-(min(axes)/max(axes))**2)))
    return [area, perimeter, 4*np.pi*area/perimeter**2 if perimeter else 0.,
            eccentricity, area/hull_area if hull_area else 0.]


def stain_features(roi, mask):
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    selected = sat[mask > 0]
    if not selected.size:
        return [float('nan'), float('nan')]
    if np.ptp(selected) == 0:
        return [0., float('nan')]
    threshold, _ = cv2.threshold(selected.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    stain = ((sat > threshold) & (mask > 0)).astype(np.uint8)*255
    fraction = float(np.count_nonzero(stain)/selected.size)
    contours, _ = cv2.findContours(stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregularity = float('nan')
    if contours:
        contour = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(contour)
        if a > 0:
            irregularity = float(cv2.arcLength(contour,True)**2/(4*np.pi*a))
    return [fraction, irregularity]


def texture_features(roi, mask):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    quant = (gray//8).astype(int)
    valid = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
    counts = np.zeros((32,32), dtype=float)
    np.add.at(counts, (quant[:, :-1][valid],quant[:, 1:][valid]), 1)
    counts += counts.T.copy()
    if counts.sum():
        p = counts/counts.sum(); i,j = np.indices(p.shape)
        mi, mj = (p*i).sum(), (p*j).sum()
        vi, vj = (p*(i-mi)**2).sum(), (p*(j-mj)**2).sum()
        corr = (p*(i-mi)*(j-mj)).sum()/np.sqrt(vi*vj) if vi*vj > 0 else 1.
        values = [np.sqrt((p*p).sum()), (p*(i-j)**2).sum(),
                  (p/(1+(i-j)**2)).sum(), -(p[p>0]*np.log2(p[p>0])).sum(), corr]
    else:
        values = [float('nan')]*5
    inner = cv2.erode(mask, np.ones((3,3),np.uint8)) > 0
    lbp = local_binary_pattern(gray,8,1,method='uniform')
    hist,_ = np.histogram(lbp[inner],bins=4,range=(0,10))
    hist = hist/hist.sum() if hist.sum() else np.zeros(4)
    return [float(v) for v in values]+hist.tolist()


def extract_all_features(contour, image):
    x,y,w,h = cv2.boundingRect(contour)
    roi=image[y:y+h,x:x+w]
    if roi.size == 0:
        raise ValueError('Contour falls outside image.')
    mask=np.zeros(roi.shape[:2],dtype=np.uint8)
    shifted=contour-np.array([[[x,y]]],dtype=contour.dtype)
    cv2.drawContours(mask,[shifted],-1,255,-1)
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    color=[]
    for channel in cv2.split(hsv):
        v=channel[mask>0]
        color.extend([float(v.mean()),float(v.std())])
    return geometric_features(contour)+stain_features(roi,mask)+color+texture_features(roi,mask)


def measure_box(image, box):
    x1,y1,x2,y2=map(int,box)
    crop=image[y1:y2,x1:x2]
    if min(crop.shape[:2],default=0)<5:
        return {'available':False,'reason':'crop_too_small'}
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    if np.ptp(gray)==0:
        return {'available':False,'reason':'constant_intensity'}
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cs=[c for c in cs if cv2.contourArea(c)>0]
    if not cs:
        return {'available':False,'reason':'no_region'}
    contour=max(cs,key=cv2.contourArea)
    values=extract_all_features(contour,crop)
    measured={name:float(v) if np.isfinite(v) else None for name,v in zip(FEATURE_NAMES,values)}
    touches=bool(np.any(contour[:,:,0]==0) or np.any(contour[:,:,1]==0) or
                 np.any(contour[:,:,0]==crop.shape[1]-1) or np.any(contour[:,:,1]==crop.shape[0]-1))
    return dict(available=True,method='largest Otsu foreground region within detector box',
                touches_crop_border=touches,foreground_regions=len(cs),features=measured)
