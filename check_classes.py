from ultralytics import YOLO

for stage in ['stage1_detection', 'stage2_classification', 'stage3_final']:
    m = YOLO(f'runs/detect/runs/{stage}/weights/best.pt')
    print(f"{stage:25s} -> {m.names}")
