import cv2
from ultralytics import YOLO
from preprocessing import preprocess_image
from segmentation import segment_image
from features import extract_all_features
from disorder_detection import detect_disorders

MODEL_PATH = 'runs/detect/runs/stage3_final/weights/best.pt'
CLASS_NAMES = ['RBC','Neutrophil','Lymphocyte',
               'Monocyte','Eosinophil','Platelet']

def analyze(image_path):
    # 1. preprocess
    processed = preprocess_image(image_path)

    # 2. segment + extract features
    contours, markers, clean = segment_image(processed)
    feature_list = []
    for contour in contours:
        fv = extract_all_features(contour, processed)
        feature_list.append({
            'area':               fv[0],
            'circularity':        fv[2],
            'nuclear_irregularity': fv[6],
            'nc_ratio':           fv[5],
            'cell_type':          'RBC'   # placeholder; YOLO overrides
        })

    # 3. YOLO detection + classification
    model = YOLO(MODEL_PATH)
    results = model(processed)

    # 4. count cells from YOLO output
    cell_counts = {n: 0 for n in CLASS_NAMES}
    for box in results[0].boxes:
        cls_idx = int(box.cls[0])
        cell_counts[CLASS_NAMES[cls_idx]] += 1

    # 5. disorder detection
    disorders = detect_disorders(cell_counts, feature_list)

    return results[0], cell_counts, disorders, feature_list

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/BCCD/train/img/BloodImage_00001.jpeg'
    result, counts, disorders, feature_list = analyze(path)
    print("Cell counts:", counts)
    print("Disorders:", disorders)
    result.save('outputs/test_output.jpg')
    print("Annotated image saved to outputs/test_output.jpg")
