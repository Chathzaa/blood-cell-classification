import sys
sys.path.append('src')
import cv2
from ultralytics import YOLO
from preprocessing import preprocess_image

img_path = 'data/BCCD_YOLO/val/images/BloodImage_00000.jpeg'   # fix path if needed

model = YOLO('runs/detect/runs/stage1_detection/weights/best.pt')

raw = cv2.imread(img_path)
pre = preprocess_image(img_path)

for name, img in [('RAW (what YOLO was trained on)', raw),
                  ('PREPROCESSED (what pipeline.py feeds it)', pre)]:
    r = model(img, verbose=False)[0]
    counts = {}
    for box in r.boxes:
        cls = model.names[int(box.cls[0])]
        counts[cls] = counts.get(cls, 0) + 1
    print(f"{name}:")
    print(f"   {counts}   (total {len(r.boxes)} boxes)")
    print()
