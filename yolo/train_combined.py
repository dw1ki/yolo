"""
Train YOLO model with combined dataset (Video 008 + WhatsApp Video)
"""
from ultralytics import YOLO

print("🚀 Training combined model (Video 008 + WhatsApp)...")
print("📊 Dataset: 977 images (798 train + 179 val)")
print("💪 Model: YOLOv8 Medium (yolov8m.pt)\n")

# Load base model
model = YOLO("yolov8m.pt")

# Train with combined dataset
results = model.train(
    data="d:/backup/pktj/backend/yolo/data/data_combined.yaml",
    epochs=60,
    imgsz=640,
    batch=4,
    device='cpu',
    project="d:/backup/pktj/backend/yolo/runs/detect",
    name="vehicle_combined",
    exist_ok=True,
    patience=10,
    save=True,
    verbose=True
)

print("\n✅ Training complete!")
print(f"📁 Model saved to: d:/backup/pktj/backend/yolo/runs/detect/vehicle_combined/weights/best.pt")
