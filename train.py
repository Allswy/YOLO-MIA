from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data='/root/VOC.yaml',
    epochs=100,
    imgsz=640,
    batch=180,
)