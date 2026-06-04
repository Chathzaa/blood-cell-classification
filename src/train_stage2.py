from ultralytics import YOLO

model = YOLO('runs/detect/runs/stage1_detection/weights/best.pt')

model.train(
    data='bloodcell.yaml',  # points to Blood Cell Images dataset
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.0005,
    freeze=10,
    project='runs',
    name='stage2_classification',
    exist_ok=True
)
print("Stage 2 complete!")