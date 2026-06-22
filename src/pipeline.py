import cv2
import numpy as np
import time
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

def get_models(log=None):
    global _model1, _model2, _model3
    if _model1 is None:
        if log: log("   Loading Stage 1 model (RBC / WBC / Platelet)...")
        _model1 = YOLO(MODEL_STAGE1)
        if log: log("   Loading Stage 2 model (WBC subtypes)...")
        _model2 = YOLO(MODEL_STAGE2)
        if log: log("   Loading Stage 3 model (Blast cell detection)...")
        _model3 = YOLO(MODEL_STAGE3)
    else:
        if log: log("   Models already loaded — using cached models.")
    return _model1, _model2, _model3


def extract_features_from_box(image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eccentricity = 0.5

    if not contours:
        w = x2 - x1
        h = y2 - y1
        area = w * h
        perimeter = 2 * (w + h)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-10)
        return {
            'area': area, 'circularity': circularity,
            'nuclear_irregularity': 1.0, 'nc_ratio': 0.5,
            'eccentricity': eccentricity,
        }

    contour = max(contours, key=cv2.contourArea)
    fv = extract_all_features(contour, crop)

    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            ma, mi  = max(ellipse[1]), min(ellipse[1])
            if ma > 0:
                eccentricity = float(np.sqrt(1 - (mi / ma) ** 2))
        except Exception:
            pass

    return {
        'area':                 fv[0],
        'circularity':          fv[2],
        'nuclear_irregularity': fv[6],
        'nc_ratio':             fv[5],
        'eccentricity':         eccentricity,
    }


def analyze(image_path, log=None):
    """
    Full pipeline. log = callable(str) used to emit progress messages.
    In Streamlit, pass a function that writes to st.status or st.empty.
    """
    def emit(msg):
        if log:
            log(msg)
        print(msg)   # always print to terminal too

    total_start = time.time()

    # ── STEP 1: Preprocessing ─────────────────────────────────────
    emit("🔵 Step 1 / 5 — Image Preprocessing")
    t0 = time.time()
    emit("   Applying Gaussian blur (noise reduction)...")
    emit("   Converting BGR → LAB color space...")
    emit("   Applying CLAHE contrast enhancement...")
    processed = preprocess_image(image_path)
    h_img, w_img = processed.shape[:2]
    emit(f"   ✅ Preprocessing done — image size: {w_img}×{h_img}px  [{time.time()-t0:.2f}s]")

    # ── STEP 2: Segmentation ──────────────────────────────────────
    emit("🔵 Step 2 / 5 — Cell Segmentation")
    t0 = time.time()
    emit("   Running Otsu's thresholding...")
    emit("   Computing distance transform...")
    emit("   Running marker-controlled Watershed algorithm...")
    emit("   Filtering contours (removing noise <100px, clumps >5000px)...")
    try:
        contours, markers, clean = segment_image(processed)
        emit(f"   ✅ Segmentation done — {len(contours)} cell regions found  [{time.time()-t0:.2f}s]")
    except Exception as e:
        contours = []
        emit(f"   ⚠️ Segmentation warning: {e}")

    # ── STEP 3: Feature Extraction ────────────────────────────────
    emit("🔵 Step 3 / 5 — Morphological Feature Extraction")
    t0 = time.time()
    emit("   Extracting geometric features (area, perimeter, circularity, eccentricity)...")
    emit("   Extracting nuclear features (N/C ratio, nuclear irregularity)...")
    emit("   Extracting texture features (GLCM, LBP)...")
    emit("   Extracting HSV color features...")

    seg_features = []
    for contour in contours:
        fv = extract_all_features(contour, processed)
        seg_features.append({
            'area': fv[0], 'circularity': fv[2],
            'nuclear_irregularity': fv[6], 'nc_ratio': fv[5],
            'eccentricity': 0.5, 'cell_type': 'Unknown'
        })
    emit(f"   ✅ Feature extraction done — {len(seg_features)} cells measured  [{time.time()-t0:.2f}s]")

    # ── STEP 4: YOLOv8 Detection ──────────────────────────────────
    emit("🔵 Step 4 / 5 — YOLOv8 Deep Learning Classification")
    t0 = time.time()

    emit("   Loading YOLOv8 models...")
    model1, model2, model3 = get_models(log=emit)

    cell_counts = {
        'RBC': 0, 'WBC': 0, 'Platelet': 0,
        'Eosinophil': 0, 'Lymphocyte': 0,
        'Monocyte': 0, 'Neutrophil': 0, 'Blast': 0,
    }

    stage1_names = {0: 'RBC', 1: 'WBC', 2: 'Platelet'}
    stage2_names = {0: 'Eosinophil', 1: 'Lymphocyte', 2: 'Monocyte', 3: 'Neutrophil'}
    stage3_names = {0: 'Normal', 1: 'Blast'}

    feature_list  = []
    rbc_features  = []
    wbc_features  = []
    blast_features= []
    wbc_crops     = []

    # Stage 1
    emit("   [Stage 1] Detecting RBC / WBC / Platelet on full slide...")
    results1 = model1(processed, verbose=False)
    for box in results1[0].boxes:
        cls_idx  = int(box.cls[0])
        cls_name = stage1_names.get(cls_idx, 'Unknown')
        if cls_name not in cell_counts:
            continue
        cell_counts[cls_name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1=max(0,x1); y1=max(0,y1); x2=min(w_img,x2); y2=min(h_img,y2)
        feats = extract_features_from_box(processed, x1, y1, x2, y2)
        if feats:
            feats['cell_type'] = cls_name
            feature_list.append(feats)
            if cls_name == 'RBC':
                rbc_features.append(feats)
            elif cls_name == 'WBC':
                wbc_features.append(feats)
                pad=10
                crop=processed[max(0,y1-pad):min(h_img,y2+pad),
                               max(0,x1-pad):min(w_img,x2+pad)]
                if crop.size > 0:
                    wbc_crops.append(crop)

    emit(f"   [Stage 1] Found — RBC: {cell_counts['RBC']}, "
         f"WBC: {cell_counts['WBC']}, Platelet: {cell_counts['Platelet']}")

    # Stage 2
    emit(f"   [Stage 2] Classifying {len(wbc_crops)} WBC crop(s) into subtypes...")
    wbc_subtype_total = 0
    for crop in wbc_crops:
        r2 = model2(crop, verbose=False)
        if r2[0].boxes and len(r2[0].boxes) > 0:
            best = max(r2[0].boxes, key=lambda b: float(b.conf[0]))
            cls_name = stage2_names.get(int(best.cls[0]), 'Unknown')
            if cls_name in cell_counts:
                cell_counts[cls_name] += 1
                wbc_subtype_total += 1
    if wbc_subtype_total > 0:
        cell_counts['WBC'] = wbc_subtype_total
    emit(f"   [Stage 2] WBC subtypes — Neutrophil: {cell_counts['Neutrophil']}, "
         f"Lymphocyte: {cell_counts['Lymphocyte']}, "
         f"Monocyte: {cell_counts['Monocyte']}, "
         f"Eosinophil: {cell_counts['Eosinophil']}")

    # Stage 3
    emit("   [Stage 3] Scanning for leukemia blast cells...")
    results3 = model3(processed, verbose=False)
    for box in results3[0].boxes:
        cls_idx  = int(box.cls[0])
        if cls_idx == 1:  # Blast
            cell_counts['Blast'] += 1
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            x1=max(0,x1); y1=max(0,y1); x2=min(w_img,x2); y2=min(h_img,y2)
            feats = extract_features_from_box(processed, x1, y1, x2, y2)
            if feats:
                feats['cell_type'] = 'Blast'
                blast_features.append(feats)
                feature_list.append(feats)
    emit(f"   [Stage 3] Blast cells detected: {cell_counts['Blast']}")
    emit(f"   ✅ YOLOv8 classification done — total cells: {sum(cell_counts.values())}  [{time.time()-t0:.2f}s]")

    # fallback if YOLO found nothing
    if not feature_list and seg_features:
        feature_list = seg_features

    # ── STEP 5: Disorder Detection ────────────────────────────────
    emit("🔵 Step 5 / 5 — Hematological Disorder Detection")
    t0 = time.time()
    emit("   Applying ALL (Leukemia) detection rules...")
    emit("   Applying Sickle Cell Disease detection rules...")
    emit("   Applying Anemia detection rules...")
    disorders = detect_disorders(
        cell_counts, feature_list,
        rbc_features=rbc_features,
        wbc_features=wbc_features,
        blast_features=blast_features,
    )
    detected = [d for d in disorders if d != 'Normal' and disorders[d].get('detected')]
    if detected:
        for d in detected:
            emit(f"   ⚠️ FLAGGED: {d} — {disorders[d]['evidence'][:80]}...")
    else:
        emit("   ✅ No disorders detected — result: Normal")
    emit(f"   ✅ Disorder detection done  [{time.time()-t0:.2f}s]")

    total_time = time.time() - total_start
    emit(f"✅ Full pipeline complete — total time: {total_time:.2f}s")

    return results1[0], cell_counts, disorders, feature_list


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_image.jpg'
    result, counts, disorders, features = analyze(path)
    print("Cell counts:", counts)
    print("Disorders:", disorders)
