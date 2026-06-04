from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # downloads automatically first time

model.train(
    data='bccd.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    cos_lr=True,
    fliplr=0.5,
    flipud=0.2,
    degrees=15,
    hsv_v=0.2,
    patience=50,
    project='runs',
    name='stage1_detection',
    exist_ok=True
)
print("Stage 1 complete!")