from ultralytics import YOLO

model = YOLO('runs/detect/runs/stage2_classification/weights/best.pt')

model.train(
    data='leukemia.yaml',  # all 3 datasets combined
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.0001,
    project='runs',
    name='stage3_final',
    exist_ok=True
)
print("Stage 3 complete! Final model ready.")