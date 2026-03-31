"""
Update API to use the combined model once training is complete
"""
import os
import shutil
import time
from pathlib import Path

train_model_path = Path("D:/backup/pktj/backend/yolo/runs/detect/vehicle_combined/weights/best.pt")
model_folder = Path("D:/backup/pktj/backend/yolo/models")
target_model = model_folder / "best_video_combined.pt"

print("⏳ Waiting for training to complete...")
print(f"📍 Watching: {train_model_path}")

# Wait for training to complete (check every 30 seconds)
while not train_model_path.exists():
    print(f"   ⏳ Training in progress... ({time.strftime('%H:%M:%S')})")
    time.sleep(30)

print(f"\n✅ Training complete! Model found.")
print(f"📋 Copying model to: {target_model}")

# Copy the trained model
shutil.copy2(train_model_path, target_model)
print(f"✅ Model copied: {target_model}")

# Update API
api_file = Path("D:/backup/pktj/backend/yolo/api.py")
with open(api_file, 'r') as f:
    content = f.read()

old_line = 'MODEL_PATH = os.path.join(BASE_DIR, "models", "best_video_008.pt")'
new_line = 'MODEL_PATH = os.path.join(BASE_DIR, "models", "best_video_combined.pt")  # ⭐ UPDATED: Combined model (008 + WhatsApp)'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(api_file, 'w') as f:
        f.write(content)
    print(f"\n✅ API updated to use combined model")
    print(f"📝 api.py line 47: {new_line}")
else:
    print(f"\n⚠️ Could not find model path line in api.py")
    print(f"   Please manually update the model path")

print(f"\n💡 Next steps:")
print(f"   1. Restart uvicorn server")
print(f"   2. Test API with both video 008 and WhatsApp video")
print(f"   3. Compare results! 🎉")
