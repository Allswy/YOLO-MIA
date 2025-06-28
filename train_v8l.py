from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="/root/VOC.yaml",
    epochs=50,
    imgsz=640,
    batch=40,
    project="/root/base_v8l",  
    name="yolov8l_voc_mia",     
    exist_ok=True              
)
