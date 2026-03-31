from ultralytics import YOLO

# Load base pretrained model
model = YOLO("yolov8m.pt")

# Train model with video 008 dataset
model.train(
    data="data.yaml",  # Using dataset_008 from video 008
    epochs=60,
    imgsz=640,  # Reduced for CPU training
    batch=2,    # Reduced for CPU training
    device='cpu',  # Using CPU (CUDA not available)
    workers=0,  # CPU training doesn't benefit from workers
    hsv_v=0.4,
    hsv_s=0.4,
    mosaic=0.2,
    patience=20,
    project="runs",
    name="train_video_008"
)

print("Training selesai! Model terbaik tersimpan di: runs/train_video_008/weights/best.pt")
