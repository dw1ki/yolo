from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # base pretrained

model.train(
    data="data.yaml",
    epochs=60,
    imgsz=1280,
    batch=4,
    device=0,
    workers=2,
    hsv_v=0.4,
    hsv_s=0.4,
    mosaic=0.2,
    patience=20
)
