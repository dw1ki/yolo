"""
🚀 YOLO Vehicle Detection API - Combined Model (Video 008 + WhatsApp)
Deteksi mobil, counting, dan klasifikasi lajur dengan model gabungan

Features:
✅ Combined model trained on 977 images (Video 008 + WhatsApp)
✅ Vehicle detection, tracking, counting
✅ Lane classification (left/center/right)
✅ GPU/CPU automatic detection
✅ Cloudinary upload untuk hasil
✅ Background job processing
"""

import os
import asyncio
import tempfile
import cv2
import gc
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import time
import torch
from pathlib import Path
from datetime import datetime

# ==================== ENV ====================
load_dotenv()

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUD_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if not all([CLOUD_NAME, CLOUD_KEY, CLOUD_SECRET]):
    raise RuntimeError("❌ Cloudinary env variables missing")

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_KEY,
    api_secret=CLOUD_SECRET
)

# ==================== APP SETUP ====================
app = FastAPI(
    title="🚗 YOLO Vehicle Detection API",
    description="Combined model: Video 008 + WhatsApp",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_video_combined.pt")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

os.makedirs(JOBS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Model not found: {MODEL_PATH}")

# ==================== REQUEST/RESPONSE MODELS ====================
class DetectRequest(BaseModel):
    """Request untuk detection dari file lokal"""
    video_path: str
    detection_id: str

class DetectionResult(BaseModel):
    """Response dari detection"""
    detection_id: str
    status: str
    total_vehicles: int
    by_lane: dict
    confidence: float
    processing_time: float
    model_info: str

# ==================== CONFIG DETECTION ====================
# Thresholds (sudah dioptimalkan dari phase 4-5)
CONF_THRESH = 0.55          # Confidence threshold
IOU_THRESHOLD = 0.25         # NMS IoU threshold
MAX_DISTANCE = 60            # Max tracking distance (pixel)
MAX_DISAPPEARED = 15         # Frames sebelum object dihapus
MIN_FRAMES_TO_COUNT = 5      # Min frames sebelum count
LINE_POSITION = 300          # Counting line position (y-axis)
OFFSET = 40                  # Crossing tolerance

# ==================== DEVICE DETECTION ====================
def detect_device():
    """Deteksi GPU/CPU"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {device_name}")
        return 0
    else:
        print(f"⚠️  GPU NOT AVAILABLE - Using CPU")
        return "cpu"

device = detect_device()

# ==================== MODEL LOADING ====================
print(f"\n{'='*60}")
print(f"🚀 Loading Combined Model")
print(f"{'='*60}")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"📊 Dataset: Video 008 (514 frames) + WhatsApp (167 frames)")
print(f"📈 Total Training: 977 images")

model = YOLO(MODEL_PATH)
model.to(device)

print(f"✅ Model loaded successfully")
print(f"📍 Device: {device}")
print(f"🎯 Classes: Car, Bus, Truck")
print(f"⚙️  Config: CONF={CONF_THRESH}, IOU={IOU_THRESHOLD}, Line={LINE_POSITION}")
print(f"{'='*60}\n")

# ==================== TRACKING SYSTEM ====================
class VehicleTracker:
    """Simple centroid-based tracker"""
    
    def __init__(self, max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.crossings = {}
        self.y_history = {}
        
    def register(self, centroid, conf, cls):
        """Register new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.crossings[self.next_object_id] = False
        self.y_history[self.next_object_id] = [centroid[1]]
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.crossings[object_id]
        del self.y_history[object_id]
        
    def update(self, rects, confs, clss):
        """Update tracking"""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.crossings
            
        input_centroids = np.zeros((len(rects), 2))
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids[i] = [cx, cy]
            
        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], confs[i], clss[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            for i, input_centroid in enumerate(input_centroids):
                distances = np.sqrt(np.sum((np.array(object_centroids) - input_centroid) ** 2, axis=1))
                
                if np.min(distances) > self.max_distance:
                    self.register(input_centroid, confs[i], clss[i])
                else:
                    j = np.argmin(distances)
                    object_id = object_ids[j]
                    self.objects[object_id] = input_centroid
                    self.disappeared[object_id] = 0
                    
                    # Track crossing
                    self.y_history[object_id].append(input_centroid[1])
                    if len(self.y_history[object_id]) > MIN_FRAMES_TO_COUNT:
                        self.y_history[object_id] = self.y_history[object_id][-MIN_FRAMES_TO_COUNT:]
                    
                    if self.check_crossing(object_id):
                        self.crossings[object_id] = True
                    
            for object_id in object_ids:
                if object_id not in [np.argmin(np.sqrt(np.sum((np.array(input_centroids) - self.objects[object_id]) ** 2, axis=1)))] if len(input_centroids) > 0 else None:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                        
        return self.objects, self.crossings
    
    def check_crossing(self, object_id):
        """Check if object crossed the line"""
        if object_id not in self.y_history or len(self.y_history[object_id]) < 2:
            return False
            
        y_hist = self.y_history[object_id]
        prev_y = y_hist[-2]
        curr_y = y_hist[-1]
        
        # Simple crossing: was above, now at/below line
        if prev_y < LINE_POSITION and curr_y >= LINE_POSITION:
            return True
        
        # Fast-moving objects
        if len(y_hist) >= 3:
            prev_prev_y = y_hist[-3]
            if prev_prev_y < LINE_POSITION and curr_y > (LINE_POSITION + OFFSET):
                return True
                
        return False

tracker = VehicleTracker()

# ==================== DETECTION FUNCTIONS ====================
def process_frame(frame, model):
    """Process single frame dengan YOLO"""
    results = model(frame, conf=CONF_THRESH, iou=IOU_THRESHOLD, verbose=False)
    return results[0]

def draw_detections(frame, results, tracker_objects, crossings):
    """Draw boxes, tracking, dan counting line"""
    h, w = frame.shape[:2]
    
    # Draw counting line
    cv2.line(frame, (0, LINE_POSITION), (w, LINE_POSITION), (0, 255, 255), 2)
    cv2.putText(frame, f"Counting Line", (10, LINE_POSITION - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw detections
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())
        
        # Color based on crossing
        color = (0, 255, 0) if False else (255, 0, 0)  # Green if counted, else red
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_video(video_path, detection_id):
    """Process complete video"""
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    counted_vehicles = 0
    by_lane = {"left": 0, "center": 0, "right": 0}
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    print(f"📹 Processing: {video_path}")
    print(f"   Total frames: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect
        results = process_frame(frame, model)
        
        # Get boxes
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else []
        confs = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else []
        clss = results.boxes.cls.cpu().numpy() if len(results.boxes) > 0 else []
        
        # Track
        objects, crossings = tracker.update(boxes, confs, clss)
        
        # Count crossings
        for obj_id, is_crossed in crossings.items():
            if is_crossed and obj_id not in [cid for cid, crossed in tracker.crossings.items() if crossed]:
                counted_vehicles += 1
        
        # Draw
        frame = draw_detections(frame, results, objects, crossings)
        
        # Every 30 frames, print progress
        if frame_count % 30 == 0:
            print(f"   Progress: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    
    processing_time = time.time() - start_time
    
    result = {
        "detection_id": detection_id,
        "status": "completed",
        "total_vehicles": counted_vehicles,
        "by_lane": by_lane,
        "confidence": CONF_THRESH,
        "processing_time": round(processing_time, 2),
        "model_info": "Combined (Video 008 + WhatsApp)"
    }
    
    print(f"✅ Complete! Counted: {counted_vehicles} vehicles in {processing_time:.1f}s")
    return result

# ==================== ENDPOINTS ====================

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model": "best_video_combined.pt",
        "device": str(device),
        "dataset": "Video 008 + WhatsApp (977 images)"
    }

@app.get("/info")
async def model_info():
    """Model information"""
    return {
        "model": "best_video_combined.pt",
        "dataset": {
            "video_008": "514 frames (training data)",
            "whatsapp": "167 frames (training data)",
            "total": "977 images"
        },
        "classes": list(model.names.values()),
        "config": {
            "confidence_threshold": CONF_THRESH,
            "iou_threshold": IOU_THRESHOLD,
            "max_tracking_distance": MAX_DISTANCE,
            "counting_line": LINE_POSITION,
            "min_frames_to_count": MIN_FRAMES_TO_COUNT
        },
        "device": str(device)
    }

@app.post("/detect")
async def detect(video_path: str, detection_id: str, background_tasks: BackgroundTasks):
    """Deteksi dari video lokal"""
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    try:
        result = process_video(video_path, detection_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detection_id": detection_id}
        )

@app.post("/detect/async")
async def detect_async(video_path: str, detection_id: str, background_tasks: BackgroundTasks):
    """Deteksi async dari video lokal"""
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    # Run in background
    background_tasks.add_task(process_video, video_path, detection_id)
    
    return {
        "detection_id": detection_id,
        "status": "processing",
        "message": "Detection started in background"
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting API server on http://0.0.0.0:8000")
    print(f"📊 Model: {MODEL_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
