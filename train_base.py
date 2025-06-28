from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="/root/VOC.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    project="/root/base",  
    name="yolov8x_voc_mia",     
    exist_ok=True               
)
