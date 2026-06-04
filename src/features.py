# import cv2
# import numpy as np
# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# def geometric_features(contour):
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)
#     circularity = (4 * np.pi * area) / (perimeter**2 + 1e-10)
#     hull = cv2.convexHull(contour)
#     hull_area = cv2.contourArea(hull) + 1e-10
#     convexity = area / hull_area
#     if len(contour) >= 5:
#         ellipse = cv2.fitEllipse(contour)
#         major = max(ellipse[1]); minor = min(ellipse[1])
#         eccentricity = np.sqrt(1 - (minor/(major+1e-10))**2)
#     else:
#         eccentricity = 0
#     return [area, perimeter, circularity, eccentricity, convexity]

# def nuclear_features(cell_roi):
#     if cell_roi.size == 0:
#         return [0, 0]
#     blue = cell_roi[:, :, 0]
#     _, nucleus = cv2.threshold(
#         blue, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )
#     nucleus_area = np.sum(nucleus > 0)
#     cell_area = cell_roi.shape[0] * cell_roi.shape[1] + 1e-10
#     nc_ratio = nucleus_area / cell_area
#     ncs, _ = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if ncs:
#         nc = max(ncs, key=cv2.contourArea)
#         np_ = cv2.arcLength(nc, True)
#         na = cv2.contourArea(nc) + 1e-10
#         irregularity = (np_**2) / (4 * np.pi * na)
#     else:
#         irregularity = 0
#     return [nc_ratio, irregularity]

# def color_features(cell_roi):
#     if cell_roi.size == 0:
#         return [0]*6
#     hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     return [h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()]

# def texture_features(cell_roi):
#     if cell_roi.size == 0:
#         return [0]*8
#     gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
#     glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
#     energy = graycoprops(glcm, 'energy')[0,0]
#     contrast = graycoprops(glcm, 'contrast')[0,0]
#     homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
#     entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
#     lbp = local_binary_pattern(gray, 8, 1, method='uniform')
#     hist, _ = np.histogram(lbp.ravel(), bins=4, range=(0,10))
#     return [energy, contrast, homogeneity, entropy] + hist.tolist()

# def extract_all_features(contour, image):
#     x, y, w, h = cv2.boundingRect(contour)
#     roi = image[y:y+h, x:x+w]
#     geo = geometric_features(contour)
#     nuc = nuclear_features(roi)
#     col = color_features(roi)
#     tex = texture_features(roi)
#     return geo + nuc + col + tex

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def geometric_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-10)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) + 1e-10
    convexity = area / hull_area
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major = max(ellipse[1]); minor = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor/(major+1e-10))**2)
    else:
        eccentricity = 0
    return [area, perimeter, circularity, eccentricity, convexity]

def nuclear_features(cell_roi):
    if cell_roi.size == 0:
        return [0, 0]
    blue = cell_roi[:, :, 0]
    _, nucleus = cv2.threshold(
        blue, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    nucleus_area = np.sum(nucleus > 0)
    cell_area = cell_roi.shape[0] * cell_roi.shape[1] + 1e-10
    nc_ratio = nucleus_area / cell_area
    ncs, _ = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ncs:
        nc = max(ncs, key=cv2.contourArea)
        np_ = cv2.arcLength(nc, True)
        na = cv2.contourArea(nc) + 1e-10
        irregularity = (np_**2) / (4 * np.pi * na)
    else:
        irregularity = 0
    return [nc_ratio, irregularity]

def color_features(cell_roi):
    if cell_roi.size == 0:
        return [0]*6
    hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()]

def texture_features(cell_roi):
    if cell_roi.size == 0:
        return [0]*8
    gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    energy = graycoprops(glcm, 'energy')[0,0]
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=4, range=(0,10))
    return [energy, contrast, homogeneity, entropy] + hist.tolist()

def extract_all_features(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]
    geo = geometric_features(contour)
    nuc = nuclear_features(roi)
    col = color_features(roi)
    tex = texture_features(roi)
    return geo + nuc + col + tex