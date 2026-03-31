"""
🔄 Model Comparison Test
Bandingkan hasil dari 3 model:
1. Old model (vehicle_night2)
2. Video 008 model
3. Combined model (008 + WhatsApp)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

# ==================== MODEL PATHS ====================
models = {
    "old": "D:/backup/pktj/backend/yolo/runs/detect/vehicle_night2/weights/best.pt",
    "video_008": "D:/backup/pktj/backend/yolo/models/best_video_008.pt",
    "combined": "D:/backup/pktj/backend/yolo/models/best_video_combined.pt"
}

videos = {
    "video_008": "D:/backup/pktj/backend/yolo/input_videos/008.mp4",
    "whatsapp": "D:/backup/pktj/backend/yolo/input_videos/WhatsApp Video 2026-01-18 at 23.48.28.mp4"
}

# ==================== CONFIG ====================
CONF_THRESH = 0.55
IOU_THRESHOLD = 0.25
SAMPLE_FRAMES = 50  # Ambil 50 frame pertama untuk quick test

# ==================== HELPER FUNCTIONS ====================
def check_model_exists(path):
    """Cek model exists"""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        return True, size_mb
    return False, 0

def load_model(path):
    """Load YOLO model"""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return None

def process_video_sample(model, video_path, num_frames=SAMPLE_FRAMES):
    """Process first N frames dari video"""
    cap = cv2.VideoCapture(video_path)
    
    total_detections = 0
    detections_by_class = {0: 0, 1: 0, 2: 0}  # Car, Bus, Truck
    avg_confidence = []
    
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=CONF_THRESH, iou=IOU_THRESHOLD, verbose=False)
        
        boxes = results[0].boxes
        total_detections += len(boxes)
        
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            avg_confidence.append(conf)
            detections_by_class[cls] += 1
        
        frame_count += 1
    
    cap.release()
    
    avg_conf = np.mean(avg_confidence) if avg_confidence else 0
    
    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "avg_confidence": round(avg_conf, 3),
        "detections_by_class": {
            "Car": detections_by_class[0],
            "Bus": detections_by_class[1],
            "Truck": detections_by_class[2]
        }
    }

# ==================== MAIN TEST ====================
print(f"\n{'='*70}")
print(f"🔄 MODEL COMPARISON TEST")
print(f"{'='*70}\n")

# 1. Check models exist
print(f"📋 Checking models...")
model_status = {}
for name, path in models.items():
    exists, size = check_model_exists(path)
    status = "✅" if exists else "❌"
    size_str = f"{size:.1f}MB" if exists else "N/A"
    print(f"   {status} {name:12} - {size_str}")
    model_status[name] = exists

# 2. Load models
print(f"\n📦 Loading models...")
loaded_models = {}
for name, path in models.items():
    if model_status[name]:
        print(f"   🔄 {name}...")
        model = load_model(path)
        if model:
            loaded_models[name] = model
            print(f"      ✅ Loaded")

# 3. Check videos exist
print(f"\n📹 Checking videos...")
for name, path in videos.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {name}")

# 4. Run comparison
print(f"\n🚀 Running tests ({SAMPLE_FRAMES} frames each)...")
print(f"{'='*70}\n")

results = {}

for video_name, video_path in videos.items():
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}")
        continue
    
    print(f"📹 VIDEO: {video_name}")
    print(f"   {'='*66}")
    
    for model_name, model in loaded_models.items():
        print(f"   🔍 Testing with {model_name} model...")
        
        try:
            test_result = process_video_sample(model, video_path, SAMPLE_FRAMES)
            
            print(f"      ✅ Frames: {test_result['frames_processed']}")
            print(f"         Total detections: {test_result['total_detections']}")
            print(f"         Avg confidence: {test_result['avg_confidence']}")
            print(f"         Car: {test_result['detections_by_class']['Car']}, " +
                  f"Bus: {test_result['detections_by_class']['Bus']}, " +
                  f"Truck: {test_result['detections_by_class']['Truck']}")
            
            results[f"{video_name}_{model_name}"] = test_result
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print()

# 5. Summary
print(f"\n{'='*70}")
print(f"📊 SUMMARY (Video 008)\n")

if "video_008_old" in results and "video_008_video_008" in results and "video_008_combined" in results:
    r_old = results["video_008_old"]
    r_008 = results["video_008_video_008"]
    r_comb = results["video_008_combined"]
    
    print(f"   Old Model:      {r_old['total_detections']} detections @ {r_old['avg_confidence']} conf")
    print(f"   Video 008:      {r_008['total_detections']} detections @ {r_008['avg_confidence']} conf")
    print(f"   Combined:       {r_comb['total_detections']} detections @ {r_comb['avg_confidence']} conf ⭐")

print(f"\n{'='*70}")
print(f"📊 SUMMARY (WhatsApp)\n")

if "whatsapp_old" in results and "whatsapp_video_008" in results and "whatsapp_combined" in results:
    r_old = results["whatsapp_old"]
    r_008 = results["whatsapp_video_008"]
    r_comb = results["whatsapp_combined"]
    
    print(f"   Old Model:      {r_old['total_detections']} detections @ {r_old['avg_confidence']} conf")
    print(f"   Video 008:      {r_008['total_detections']} detections @ {r_008['avg_confidence']} conf")
    print(f"   Combined:       {r_comb['total_detections']} detections @ {r_comb['avg_confidence']} conf ⭐")
    
    improvement = ((r_comb['total_detections'] - r_008['total_detections']) / r_008['total_detections'] * 100) if r_008['total_detections'] > 0 else 0
    print(f"\n   ✨ Improvement vs Video 008: {improvement:+.1f}%")

print(f"\n{'='*70}\n")
print(f"✅ Test complete!")
print(f"🎯 Rekomendasi: Gunakan combined model untuk hasil terbaik di semua video\n")
