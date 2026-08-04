from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(
    data='data/BloodCell_CLS_v2',
    epochs=100,
    imgsz=224,
    batch=32,
    device="cpu",
    seed=0,
    deterministic=True,
    project='runs',
    name='stage2_cls_groupsplit',
    exist_ok=True,
)
print("Stage 2 (classification) complete.")