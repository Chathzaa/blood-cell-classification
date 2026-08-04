import sys; sys.path.append('src')
import cv2
from pipeline import get_models

model1, model2, model3 = get_models()
print("Stage 3 classes:", model3.names)

test_images = [
    'data/BCCD_YOLO/val/images/BloodImage_00000.jpeg',
    'data/BCCD_YOLO/val/images/BloodImage_00002.jpeg',
    'data/BCCD_YOLO/val/images/BloodImage_00014.jpeg',
]

for img_path in test_images:
    raw = cv2.imread(img_path)
    if raw is None:
        print(f"Could not read {img_path}")
        continue
    results1 = model1(raw, verbose=False)
    for box in results1[0].boxes:
        cls_idx = int(box.cls[0])
        if cls_idx == 1:  # WBC
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad = 10
            crop = raw[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
            r3 = model3(crop, verbose=False)
            print(f"\n{img_path}")
            print("  Probs (Benign/Early/Pre/Pro):", [round(p,3) for p in r3[0].probs.data.tolist()])
            print("  Top1:", model3.names[r3[0].probs.top1], f"({r3[0].probs.top1conf.item():.3f})")