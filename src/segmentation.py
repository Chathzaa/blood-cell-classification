# import cv2
# import numpy as np

# def otsu_threshold(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(
#         gray, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )
#     return binary

# def distance_transform(binary):
#     dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#     cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
#     return dist

# def watershed_segment(original, binary, dist):
#     _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
#     sure_fg = np.uint8(sure_fg)
#     sure_bg = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations=3)
#     unknown = cv2.subtract(sure_bg, sure_fg)
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0
#     markers = cv2.watershed(original, markers)
#     return markers

# def morphological_cleanup(binary):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
#     return closed

# def filter_contours(binary):
#     contours, _ = cv2.findContours(
#         binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     valid = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
#     return valid

# def segment_image(processed_image, max_size=320):
#     # RESIZE first — this is what stops the freeze
#     h, w = processed_image.shape[:2]
#     if max(h, w) > max_size:
#         scale = max_size / max(h, w)
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         image = cv2.resize(processed_image, (new_w, new_h))
#     else:
#         image = processed_image.copy()

#     binary  = otsu_threshold(image)
#     clean   = morphological_cleanup(binary)
#     dist    = distance_transform(clean)
#     markers = watershed_segment(image, clean, dist)
#     contours = filter_contours(clean)
#     return contours, markers, clean, image

import cv2
import numpy as np

def otsu_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary

def distance_transform(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    return dist

def watershed_segment(original, binary, dist):
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(original, markers)
    return markers

def morphological_cleanup(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def filter_contours(binary):
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
    return valid

def segment_image(processed_image):
    binary = otsu_threshold(processed_image)
    dist = distance_transform(binary)
    markers = watershed_segment(processed_image, binary, dist)
    clean = morphological_cleanup(binary)
    contours = filter_contours(clean)
    return contours, markers, clean