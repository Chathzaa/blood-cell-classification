import cv2
import numpy as np
from ultralytics import YOLO
from preprocessing import preprocess_image
from segmentation import segment_image
from features import extract_all_features
from disorder_detection import detect_disorders

MODEL_STAGE1 = 'runs/detect/runs/stage1_detection/weights/best.pt'
MODEL_STAGE2 = 'runs/detect/runs/stage2_classification/weights/best.pt'
MODEL_STAGE3 = 'runs/detect/runs/stage3_final/weights/best.pt'

_model1 = None
_model2 = None
_model3 = None

def get_models():
    global _model1, _model2, _model3
    if _model1 is None:
        _model1 = YOLO(MODEL_STAGE1)
        _model2 = YOLO(MODEL_STAGE2)
        _model3 = YOLO(MODEL_STAGE3)
    return _model1, _model2, _model3


def extract_features_from_box(image, x1, y1, x2, y2):
    """Extract morphological features from a YOLO bounding box region."""
    # crop the cell region
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    # convert to grayscale and threshold to get contour
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # fallback: use full box as contour
        w = x2 - x1
        h = y2 - y1
        area = w * h
        perimeter = 2 * (w + h)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-10)
        return {
            'area': area,
            'circularity': circularity,
            'nuclear_irregularity': 1.0,
            'nc_ratio': 0.5,
        }

    # use largest contour
    contour = max(contours, key=cv2.contourArea)
    fv = extract_all_features(contour, crop)

    return {
        'area':                 fv[0],
        'circularity':          fv[2],
        'nuclear_irregularity': fv[6],
        'nc_ratio':             fv[5],
    }


def analyze(image_path):
    # 1. Preprocess
    processed = preprocess_image(image_path)
    h_img, w_img = processed.shape[:2]

    # 2. Load models
    model1, model2, model3 = get_models()

    # 3. Stage 1 — detect RBC, WBC, Platelet
    results1 = model1(processed)
    stage1_names = {0: 'RBC', 1: 'WBC', 2: 'Platelet'}

    cell_counts = {
        'RBC': 0, 'WBC': 0, 'Platelet': 0,
        'Eosinophil': 0, 'Lymphocyte': 0,
        'Monocyte': 0, 'Neutrophil': 0, 'Blast': 0,
    }

    # feature lists per cell type — built from YOLO boxes directly
    feature_list = []   # all cells for morphological summary in UI
    rbc_features  = []  # only RBCs — for anemia + sickle cell
    wbc_features  = []  # only WBCs — for ALL

    wbc_crops = []  # WBC crops for Stage 2

    for box in results1[0].boxes:
        cls_idx  = int(box.cls[0])
        cls_name = stage1_names.get(cls_idx, 'Unknown')
        if cls_name not in cell_counts:
            continue

        cell_counts[cls_name] += 1

        # extract features from this box region
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)

        feats = extract_features_from_box(processed, x1, y1, x2, y2)
        if feats:
            feats['cell_type'] = cls_name
            feature_list.append(feats)

            if cls_name == 'RBC':
                rbc_features.append(feats)
            elif cls_name == 'WBC':
                wbc_features.append(feats)
                # save crop for Stage 2
                pad = 10
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                cx2 = min(w_img, x2 + pad); cy2 = min(h_img, y2 + pad)
                crop = processed[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    wbc_crops.append(crop)

    # 4. Stage 2 — classify each WBC crop into subtypes
    stage2_names = {0: 'Eosinophil', 1: 'Lymphocyte', 2: 'Monocyte', 3: 'Neutrophil'}
    wbc_subtype_total = 0

    for crop in wbc_crops:
        result2 = model2(crop, verbose=False)
        if result2[0].boxes and len(result2[0].boxes) > 0:
            best_box = max(result2[0].boxes, key=lambda b: float(b.conf[0]))
            cls_idx  = int(best_box.cls[0])
            cls_name = stage2_names.get(cls_idx, 'Unknown')
            if cls_name in cell_counts:
                cell_counts[cls_name] += 1
                wbc_subtype_total += 1

    if wbc_subtype_total > 0:
        cell_counts['WBC'] = wbc_subtype_total

    # 5. Stage 3 — detect Blast cells
    results3 = model3(processed, verbose=False)
    blast_features = []
    for box in results3[0].boxes:
        cls_idx  = int(box.cls[0])
        if cls_idx == 1:  # Blast
            cell_counts['Blast'] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            feats = extract_features_from_box(processed, x1, y1, x2, y2)
            if feats:
                feats['cell_type'] = 'Blast'
                blast_features.append(feats)
                feature_list.append(feats)

    # 6. Segmentation — only for morphological display in UI
    # (not used for disorder detection anymore)
    try:
        contours, markers, clean = segment_image(processed)
        if not feature_list:
            # fallback if YOLO found nothing
            for contour in contours:
                fv = extract_all_features(contour, processed)
                feature_list.append({
                    'area': fv[0], 'circularity': fv[2],
                    'nuclear_irregularity': fv[6], 'nc_ratio': fv[5],
                    'cell_type': 'Unknown'
                })
    except Exception:
        pass

    # 7. Disorder detection using reliable per-cell-type features
    disorders = detect_disorders(
        cell_counts, feature_list,
        rbc_features=rbc_features,
        wbc_features=wbc_features,
        blast_features=blast_features,
    )

    return results1[0], cell_counts, disorders, feature_list


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_image.jpg'
    result, counts, disorders, features = analyze(path)
    print("Cell counts:", counts)
    print("Disorders:", disorders)
