import cv2
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path)
    return img

def apply_gaussian(image, sigma=1.5):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def rgb_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def apply_clahe(lab_image, clip_limit=3.5, tile_size=8):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    l, a, b = cv2.split(lab_image)
    l_enhanced = clahe.apply(l)
    return cv2.merge([l_enhanced, a, b])

def preprocess_image(image_path):
    img = load_image(image_path)
    blurred = apply_gaussian(img)
    lab = rgb_to_lab(blurred)
    enhanced_lab = apply_clahe(lab)
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return result