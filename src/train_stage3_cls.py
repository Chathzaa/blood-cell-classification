from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(
    data='data/Leukemia_CLS_v2',
    epochs=100,
    imgsz=224,
    batch=32,
    device="cpu",
    seed=0,
    deterministic=True,
    project='runs',
    name='stage3_cls_dedup',
    exist_ok=True,
)
print("Stage 3 (classification) complete.")