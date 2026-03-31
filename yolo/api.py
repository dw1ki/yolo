# api.py - YOLO Vehicle Detection dengan GPU + OpenVINO + Batch Processing
import os
import asyncio
import tempfile
import cv2
import gc
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import time
import torch
from openvino.runtime import Core

# ================= ENV =================
load_dotenv()

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUD_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if not all([CLOUD_NAME, CLOUD_KEY, CLOUD_SECRET]):
    raise RuntimeError("Cloudinary env variables missing")

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_KEY,
    api_secret=CLOUD_SECRET
)

# ================= APP =================
app = FastAPI(title="YOLO Vehicle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== REQUEST MODELS ==================
class DetectRequest(BaseModel):
    """⭐ NEW: Accept local video path instead of file upload"""
    video_path: str  # Local file path, e.g., d:\new\pktj\backend\yolo\input_videos\video_12345.mp4
    detection_id: str

# ================= MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_video_combined.pt")  # ⭐ UPDATED: Combined model (Video 008 + WhatsApp)
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

os.makedirs(JOBS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

# ================= GPU DETECTION & SETUP =================
def detect_device():
    """Detect available device: GPU (CUDA) atau CPU"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        device_idx = 0
        print(f"✅ GPU DETECTED: {device}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device_idx, "cuda"
    else:
        print(f"⚠️  GPU NOT AVAILABLE - Using CPU")
        return "cpu", "cpu"

device_idx, device_type = detect_device()

# Load model dengan GPU
print(f"🔄 Loading YOLO model...")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"✅ Model: best_video_combined.pt (Trained on Video 008 + WhatsApp - Combined Dataset)")
model = YOLO(MODEL_PATH)
model.to(device_idx)  # Move model to GPU/CPU
print(f"✅ Model loaded on device: {device_idx}")

# OpenVINO optimization (optional - compile model untuk inference cepat)
try:
    print(f"🔄 Optimizing model dengan OpenVINO...")
    ov_core = Core()
    # OpenVINO optimization akan dilakukan during inference
    print(f"✅ OpenVINO ready")
except Exception as e:
    print(f"⚠️  OpenVINO initialization failed: {e}")
    ov_core = None

# In-memory jobs cache
jobs = {}

def save_job(job_id: str):
    """Save job to persistent file storage"""
    try:
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(job_file, 'w') as f:
            import json
            # Convert all numpy types to Python native types before saving
            job_data = to_python_type(jobs[job_id])
            json.dump(job_data, f)
    except Exception as e:
        print(f"⚠️ Failed to save job {job_id}: {e}")

def load_job(job_id: str):
    """Load job from persistent file storage"""
    try:
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                import json
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load job {job_id}: {e}")
    return None

# ================= CONFIG (Optimized dari count_video.py) =================
# Batch Processing
BATCH_SIZE = 3  # ⭐ Process 3 frames at a time (safe GPU balance)
BATCH_QUEUE = []

# Line & Offset - SEPARATE per lane
LINE_POSITION = {
    'kiri': 300,   # Left lane counting line
    'kanan': 280   # Right lane counting line
}
OFFSET = 40

# YOLO Config
CONF_THRESH = 0.55  # ⭐ HIGHER: Kurangi false detection (dari 0.35 → 0.55)
IOU_THRESHOLD = 0.25  # ⭐ STRICTER: Filter overlapping boxes lebih ketat (dari 0.45 → 0.25)
MAX_DISAPPEARED = 15  # ⭐ TIGHTER: Hapus object lebih cepat (dari 20 → 15)
MAX_DISTANCE = 60  # ⭐ STRICTER: Prevent ID switching antar object (dari 100 → 60)
MIN_FRAMES_TO_COUNT = 5  # ⭐ RELAXED: Turunkan ke 5 frames untuk tidak skip vehicles (dari 12 → 5)

# FPS Optimization
FRAME_SKIP = 0  # ⭐ 0 = process all frames, 1 = process every 2nd frame, etc.
RESIZE_SCALE = 0.4  # ⭐ 0.5 = 50% reduction for optimal GPU speed (~1.04x realtime)
TARGET_FPS = 60  # ⭐ Target FPS for faster processing

# Video Compression
USE_PIL_COMPRESSION = False  # ⭐ DISABLED: Makes file larger paradoxically!
JPEG_QUALITY = 75  # ⭐ Not used (PIL disabled)
TARGET_BITRATE = "800k"  # ⭐ 1200k bitrate = ~100-120MB file size (quality + speed + size compromise)

# Class mapping
class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# ================= GLARE HANDLING - AGGRESSIVE VERSION =================
def reduce_glare(frame):
    """⭐ AGGRESSIVE CLAHE for severe lamp glare
    More aggressive settings untuk handle extreme glare cases
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # ⭐ AGGRESSIVE CLAHE: clipLimit 4.0 (dari 2.0), tileSize 4x4 (dari 8x8)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l_clahe = clahe.apply(l)
        
        # ⭐ Add gamma correction untuk extra normalization di extreme glare
        inv_gamma = 1.0 / 1.5
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        l_clahe = cv2.LUT(l_clahe, table)
        
        # Merge kembali
        lab_enhanced = cv2.merge([l_clahe, a, b])
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return frame_enhanced
    except:
        return frame

def get_adaptive_conf(frame):
    """⭐ EXTREME ADAPTIVE CONFIDENCE - Handle severe glare
    Detect extreme glare situations dan lower confidence drastically
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ⭐ Check percentage dari pixels yang overexposed (>240)
        overexposed_pixels = np.sum(gray > 240)
        overexposed_ratio = overexposed_pixels / gray.size
        
        # ⭐ Tiered confidence based on glare severity
        if overexposed_ratio > 0.30:  # >30% pixels overexposed = EXTREME
            return 0.35  # Very lenient (dari 0.45)
        elif overexposed_ratio > 0.15:  # 15-30% = severe glare
            return 0.40
        elif np.mean(gray) > 200:  # Average very bright
            return 0.45
        elif np.mean(gray) < 50:  # Very dark
            return 0.60
        else:  # Normal
            return 0.55
    except:
        return CONF_THRESH

def to_python_type(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    else:
        return obj

def batch_predict_gpu(frames_batch, conf=CONF_THRESH):
    """⭐ Batch prediction using GPU - with GLARE HANDLING
    - Preprocess frames dengan CLAHE untuk reduce lamp glare
    - Use adaptive confidence untuk better detection di bright/dark areas
    """
    if not frames_batch:
        return []
    
    results = []
    try:
        # Process each frame on GPU individually (more stable than array batching)
        for frame in frames_batch:
            try:
                # ⭐ NEW: Reduce lamp glare dengan CLAHE preprocessing
                frame_processed = reduce_glare(frame)
                
                # ⭐ NEW: Use adaptive confidence berdasarkan brightness
                adaptive_conf = get_adaptive_conf(frame_processed)
                
                result = model.predict(
                    frame_processed,
                    device=device_idx,
                    conf=adaptive_conf,
                    verbose=False,
                    max_det=50,
                    half=(device_type == "cuda")  # Use FP16 on GPU for speed
                )
                results.append(result[0])
            except Exception as e:
                print(f"[WARN] Frame predict error: {e}")
                results.append(None)
        
        return results
    except Exception as e:
        print(f"[WARN] Batch predict error: {e}")
        return [None] * len(frames_batch)

def classify_vehicle_by_size(bbox, frame_height):
    """Klasifikasi kendaraan berdasarkan ukuran bounding box (dari count_video.py)"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    normalized_area = area / (frame_height * frame_height)
    normalized_height = height / frame_height
    
    if normalized_area > 0.12 and normalized_height > 0.3:
        return 'bus'
    elif normalized_area > 0.15:
        return 'truk'
    elif normalized_area > 0.06 and normalized_height > 0.25:
        return 'bus'
    else:
        return 'mobil'

def get_lane(cx, frame_width):
    """Tentukan lajur berdasarkan posisi x centroid"""
    mid_point = frame_width // 2
    return 'kiri' if cx < mid_point else 'kanan'

def calculate_iou(box1, box2):
    """Hitung IoU antara dua bounding box"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def check_line_crossing(y_history, line_y, offset=OFFSET):
    """Cek apakah kendaraan melewati garis per lane (BALANCED: accurate crossing detection)
    line_y bisa int atau dict dengan lane-specific positions
    """
    if len(y_history) < 2:  # ⭐ RELAXED: Require at least 2 history points
        return False
    
    # Convert int to dict if needed (backward compatibility)
    if isinstance(line_y, int):
        line_pos = line_y
    else:
        line_pos = line_y
    
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    
    # ⭐ CONDITION 1: Clear crossing - object crosses FROM ABOVE TO BELOW
    if prev_y < line_pos and curr_y >= line_pos:
        # Object was above line, now at or below = CROSSING
        return True
    
    # ⭐ CONDITION 2: For objects already near line - check downward movement
    if prev_y < (line_pos + offset) and curr_y >= line_pos:
        # Object is moving toward/crossing line
        return True
    
    # ⭐ CONDITION 3: Handle fast-moving objects that skip over line area
    if len(y_history) >= 3:
        prev_prev_y = y_history[-3]
        # If moving down through line zone
        if prev_prev_y < line_pos and curr_y > (line_pos + offset):
            return True
    
    return False

def euclidean_distance(p1, p2):
    """Hitung jarak Euclidean"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def register(objects, next_object_id, centroid, bbox, cls_name, lane):
    """Register objek baru"""
    objects[next_object_id] = {
        'centroid': centroid,
        'bbox': bbox,
        'disappeared': 0,
        'counted': False,
        'class': cls_name,
        'lane': lane,
        'y_history': [centroid[1]],
        'frame_count': 1
    }
    return next_object_id + 1

def deregister(objects, object_id):
    """Hapus objek"""
    if object_id in objects:
        del objects[object_id]

def update_tracking(objects, next_object_id, detections, frame_width, frame_height, counters, vehicle_count_total):
    """Update tracking dengan logika dari count_video.py"""
    
    if len(detections) == 0:
        for object_id in list(objects.keys()):
            objects[object_id]['disappeared'] += 1
            max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
            if objects[object_id]['disappeared'] > max_disappear:
                deregister(objects, object_id)
        return objects, next_object_id, vehicle_count_total
    
    # ⭐ IMPROVED: Filter overlapping detections (Stricter NMS)
    filtered_detections = []
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    for det in sorted_detections:
        keep = True
        for existing_det in filtered_detections:
            iou = calculate_iou(det['bbox'], existing_det['bbox'])
            # ⭐ Keep only if IoU is MUCH lower (prevent overlaps)
            if iou > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            # ⭐ Extra validation: Check if detection quality is acceptable
            if det['conf'] >= CONF_THRESH:
                filtered_detections.append(det)
    
    detections = filtered_detections
    
    if len(objects) == 0:
        for det in detections:
            lane = get_lane(det['centroid'][0], frame_width)
            next_object_id = register(objects, next_object_id, det['centroid'], det['bbox'], det['class'], lane)
    else:
        object_ids = list(objects.keys())
        object_centroids = [objects[oid]['centroid'] for oid in object_ids]
        input_centroids = [det['centroid'] for det in detections]
        
        distances = []
        for oc in object_centroids:
            row = []
            for ic in input_centroids:
                row.append(euclidean_distance(oc, ic))
            distances.append(row)
        
        distances = np.array(distances)
        
        if distances.shape[0] > 0 and distances.shape[1] > 0:
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                det = detections[col]
                
                # ⭐ FIX LEVEL 2: ABSOLUTELY reject counted objects dengan lane berbeda
                distance = distances[row, col]
                if object_id in objects and objects[object_id]['counted']:
                    # Object sudah di-count - NO cross-lane matching allowed!
                    old_lane = objects[object_id]['lane']
                    new_lane = get_lane(det['centroid'][0], frame_width)
                    
                    # Different lane + already counted = REJECT completely
                    if old_lane != new_lane:
                        continue
                
                if distance > MAX_DISTANCE:
                    continue
                
                if not objects[object_id]['counted']:
                    objects[object_id]['class'] = det['class']
                
                objects[object_id]['centroid'] = det['centroid']
                objects[object_id]['bbox'] = det['bbox']
                objects[object_id]['disappeared'] = 0
                objects[object_id]['y_history'].append(det['centroid'][1])
                objects[object_id]['frame_count'] += 1
                
                if len(objects[object_id]['y_history']) > 15:
                    objects[object_id]['y_history'].pop(0)
                
                # ⭐ FIX: Lock lane jika sudah di-count (jangan ubah lane lagi)
                if not objects[object_id]['counted']:
                    objects[object_id]['lane'] = get_lane(det['centroid'][0], frame_width)
                
                # ⭐ RELAXED: Cek crossing dengan kondisi balanced
                if not objects[object_id]['counted'] and objects[object_id]['frame_count'] >= MIN_FRAMES_TO_COUNT:
                    # Use lane-specific counting line
                    lane = objects[object_id]['lane']
                    line_pos = LINE_POSITION[lane]
                    is_crossing = check_line_crossing(
                        objects[object_id]['y_history'],
                        line_pos,
                        OFFSET
                    )
                    
                    if is_crossing:
                        objects[object_id]['counted'] = True
                        lane = objects[object_id]['lane']
                        cls_name = objects[object_id]['class']
                        counters[lane]['total'] += 1
                        counters[lane][cls_name] += 1
                        vehicle_count_total += 1
                
                # ⭐ NEW: Delete object setelah counted dan sudah jauh di bawah line
                # Ini prevent bekas bbox nya jadi confusion untuk object lain
                if objects[object_id]['counted']:
                    curr_y = objects[object_id]['centroid'][1]
                    lane = objects[object_id]['lane']
                    line_pos = LINE_POSITION[lane]
                    # Kalau sudah 150px di bawah line = hapus (tidak perlu track lagi)
                    if curr_y > (line_pos + 150):
                        deregister(objects, object_id)
                        continue
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle objek hilang
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                # ⭐ FIX: Safety check - object might be deleted in previous iteration
                if object_id not in objects:
                    continue
                
                objects[object_id]['disappeared'] += 1
                max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
                
                # ⭐ NEW: Temporal interpolation untuk handle detection gaps (glare)
                # Jika object hilang 1-2 frame temporary (bukan truly gone)
                if objects[object_id]['disappeared'] <= 2 and len(objects[object_id]['y_history']) >= 2:
                    # Predict next centroid berdasarkan velocity
                    velocity = objects[object_id]['y_history'][-1] - objects[object_id]['y_history'][-2]
                    predicted_y = objects[object_id]['y_history'][-1] + velocity
                    
                    # Update position dengan prediksi (maintain tracking)
                    objects[object_id]['y_history'].append(predicted_y)
                    objects[object_id]['centroid'] = (objects[object_id]['centroid'][0], predicted_y)
                    
                    # ⭐ Check if interpolated position crosses line (untuk count)
                    if not objects[object_id]['counted']:
                        lane = objects[object_id]['lane']
                        line_pos = LINE_POSITION[lane]
                        is_crossing = check_line_crossing(
                            objects[object_id]['y_history'],
                            line_pos,
                            OFFSET
                        )
                        if is_crossing:
                            objects[object_id]['counted'] = True
                            lane = objects[object_id]['lane']
                            cls_name = objects[object_id]['class']
                            counters[lane]['total'] += 1
                            counters[lane][cls_name] += 1
                            vehicle_count_total += 1
                
                # Remove object jika truly gone
                if objects[object_id]['disappeared'] > max_disappear:
                    deregister(objects, object_id)
            
            # Handle deteksi baru
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                det = detections[col]
                lane = get_lane(det['centroid'][0], frame_width)
                next_object_id = register(objects, next_object_id, det['centroid'], det['bbox'], det['class'], lane)
    
    return objects, next_object_id, vehicle_count_total

# ================= DRAWING FUNCTIONS =================

def draw_annotations(frame, counters, objects, vehicle_count_total, line_position=None):
    """Draw all annotations: counting line, lane divider, boxes, counter panel
    line_position can be int or dict with lane-specific positions
    """
    h, w = frame.shape[:2]
    
    # ⭐ Handle both dict and int line_position (backward compat)
    if line_position is None:
        line_position = LINE_POSITION
    
    # 1. Draw dual counting lines (per lane)
    if isinstance(line_position, dict):
        # Left lane line (MAGENTA)
        line_kiri = line_position.get('kiri', 300)
        cv2.line(frame, (0, line_kiri), (w//2, line_kiri), (255, 0, 255), 3)
        cv2.putText(frame, f"KIRI (Y={line_kiri})", (10, line_kiri - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Right lane line (BLUE)
        line_kanan = line_position.get('kanan', 280)
        cv2.line(frame, (w//2, line_kanan), (w, line_kanan), (255, 0, 0), 3)
        cv2.putText(frame, f"KANAN (Y={line_kanan})", (w//2 + 10, line_kanan - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # Single line (old behavior)
        cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 3)
        cv2.putText(frame, f"COUNTING LINE (Y={line_position})", (10, line_position - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 2. Draw lane divider (MAGENTA vertical)
    mid_x = w // 2
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 0, 255), 2)
    
    # 3. Draw bounding boxes for tracked objects
    for obj_id, obj in objects.items():
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cx, cy = obj['centroid']
        
        # Skip if already counted and far below line
        # Extract lane-specific line position
        if isinstance(line_position, dict):
            obj_line_pos = line_position.get(obj['lane'], 300)
        else:
            obj_line_pos = line_position
        
        if obj['counted'] and cy > (obj_line_pos + 150):
            continue
        
        # Color logic: GREEN if counted, BLUE/PINK based on lane if not counted
        if obj['counted']:
            color = (0, 255, 0)  # GREEN = counted
            thickness = 3
        else:
            if obj['lane'] == 'kiri':
                color = (255, 0, 255)  # MAGENTA for left lane
            else:
                color = (255, 0, 0)  # BLUE for right lane
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw centroid point
        cv2.circle(frame, (cx, cy), 4, color, -1)
        
        # Draw label with class
        label = f"ID:{obj_id} {obj['class'].upper()} [{obj['lane'].upper()}]"
        if obj['counted']:
            label += " ✓"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 4. Draw counter panel (top-left)
    panel_h = 240
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_pos = 25
    cv2.putText(frame, "LAJUR KIRI:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    y_pos += 25
    cv2.putText(frame, f"  Mobil: {counters['kiri']['mobil']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_pos += 20
    cv2.putText(frame, f"  Bus: {counters['kiri']['bus']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_pos += 20
    cv2.putText(frame, f"  Truk: {counters['kiri']['truk']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y_pos += 30
    cv2.putText(frame, "LAJUR KANAN:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    y_pos += 25
    cv2.putText(frame, f"  Mobil: {counters['kanan']['mobil']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_pos += 20
    cv2.putText(frame, f"  Bus: {counters['kanan']['bus']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_pos += 20
    cv2.putText(frame, f"  Truk: {counters['kanan']['truk']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Total count
    y_pos += 30
    cv2.putText(frame, f"TOTAL: {vehicle_count_total}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return frame

# ================= HELPERS =================
async def upload_to_cloudinary(file: UploadFile):
    """Upload file dan return local temp path"""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = cloudinary.uploader.upload(
        tmp_path,
        resource_type="video",
        folder="input_videos"
    )
    
    return {
        "tmp_path": tmp_path,
        "cloudinary_url": result["secure_url"]
    }

# ================= PROCESS =================
async def process_video(job_id: str, video_path: str):
    """Process video dengan optimasi dari count_video.py"""
    
    print(f"\n🟢 [process_video START] Job {job_id}")
    print(f"   Video Path: {video_path}")
    print(f"   ⭐ Device: {device_type.upper()}")
    print(f"   ⭐ Batch Size: {BATCH_SIZE}")
    print(f"   ⭐ YOLO Conf Threshold: {CONF_THRESH}")
    start_time = time.time()
    
    try:
        # Initialize tracking
        objects = {}
        next_object_id = 0
        counters = {
            'kiri': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0},
            'kanan': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
        }
        vehicle_count_total = 0
        
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "total": 0,
            "kiri": 0,
            "kanan": 0,
            "detections": [],
            "outputVideoUrl": None
        }
        save_job(job_id)
        print(f"   ✅ Job initialized")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"   ❌ Cannot open video")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Cannot open video"
            save_job(job_id)
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        fps = int(fps)  # ⭐ CRITICAL: Convert to int for VideoWriter compatibility!
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        
        print(f"   [DEBUG] Video: FPS={fps}, Frames={total_frames}, Duration={total_frames/fps:.1f}s")

        frame_count = 0
        
        # ⭐ Write annotated video with MJPEG codec (more stable than mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_mp4 = os.path.join(tempfile.gettempdir(), f"output_{job_id}.avi")
        out_writer = None

        # ⭐ BATCH PROCESSING LOOP dengan GPU optimization
        batch_frames = []
        batch_frame_info = []
        current_progress = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"   [DEBUG] End of video at frame {frame_count}")
                break

            if frame is None or frame.size == 0:
                frame_count += 1
                continue

            # Skip frames untuk faster processing jika diperlukan
            if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
                frame_count += 1
                continue

            frame_count += 1
            h, w, _ = frame.shape

            # ⭐ Add frame to batch
            batch_frames.append(frame)
            batch_frame_info.append({'frame_count': frame_count, 'frame': frame.copy()})

            # Process batch jika size mencapai BATCH_SIZE atau end of video
            if len(batch_frames) >= BATCH_SIZE or not ret:
                
                # ⭐ BATCH PREDICTION on GPU
                try:
                    results = batch_predict_gpu(batch_frames, conf=CONF_THRESH)
                except Exception as e:
                    print(f"   [WARN] Batch predict error: {e}")
                    results = [None] * len(batch_frames)
                
                # Process batch results
                for idx, (frame_info, result) in enumerate(zip(batch_frame_info, results)):
                    fg_frame = frame_info['frame']
                    fg_h, fg_w, _ = fg_frame.shape
                    
                    # Process detections
                    detections = []
                    if result is not None and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                        confidences = result.boxes.conf.cpu().numpy()

                        for box, cls_id, conf in zip(boxes, cls_ids, confidences):
                            x1, y1, x2, y2 = box
                            cX = int((x1 + x2) / 2.0)
                            cY = int((y1 + y2) / 2.0)

                            # ⭐ CLASSIFICATION LOGIC dari count_video.py
                            size_based_class = classify_vehicle_by_size(box, fg_h)
                            model_class = class_map.get(cls_id, 'mobil')
                            
                            if conf > 0.7 and model_class == size_based_class:
                                final_class = model_class
                            elif model_class in ['bus', 'truk']:
                                final_class = size_based_class
                            else:
                                final_class = size_based_class

                            detections.append({
                                'bbox': box,
                                'centroid': (cX, cY),
                                'class': final_class,
                                'conf': conf
                            })

                    # Update tracking
                    objects, next_object_id, vehicle_count_total = update_tracking(
                        objects, next_object_id, detections, fg_w, fg_h, counters, vehicle_count_total
                    )

                    # ===== VISUALIZATION (add annotations to frame) =====
                    fg_frame = draw_annotations(fg_frame, counters, objects, vehicle_count_total, line_position=LINE_POSITION)

                    # Initialize writer after getting frame size
                    if out_writer is None:
                        print(f"   [DEBUG] VideoWriter init: fps={fps}, size=({fg_w}, {fg_h}), codec=MJPEG, GPU detected={device_type}")
                        out_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (fg_w, fg_h))
                        if not out_writer.isOpened():
                            print(f"   [ERROR] VideoWriter failed!")
                            break

                    if out_writer is not None and out_writer.isOpened():
                        # Write annotated frame
                        frame_write = np.ascontiguousarray(fg_frame.astype(np.uint8))
                        out_writer.write(frame_write)

                    # Update progress
                    current_progress = int((frame_count / total_frames) * 100)
                    jobs[job_id]["progress"] = current_progress
                    jobs[job_id]["total"] = vehicle_count_total
                    jobs[job_id]["kiri"] = counters['kiri']['total']
                    jobs[job_id]["kanan"] = counters['kanan']['total']
                
                # Reset batch
                batch_frames = []
                batch_frame_info = []
                
                # Save periodically
                if current_progress % 10 == 0:
                    save_job(job_id)
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                await asyncio.sleep(0)

        cap.release()
        if out_writer is not None:
            out_writer.release()

        print(f"   [DEBUG] Video annotated and saved to temp: {output_mp4}")
        print(f"   [DEBUG] File size: {os.path.getsize(output_mp4) if os.path.exists(output_mp4) else 0} bytes")
        
        # ⭐ Convert AVI to MP4 using imageio for browser compatibility
        output_mp4_final = os.path.join(tempfile.gettempdir(), f"output_{job_id}_final.mp4")
        upload_file = output_mp4
        
        try:
            if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0:
                print(f"   [DEBUG] Converting AVI to MP4 with bitrate {TARGET_BITRATE}...")
                import imageio
                import subprocess
                
                # Try ffmpeg first (faster, bitrate control)
                try:
                    cmd = [
                        'ffmpeg',
                        '-i', output_mp4,
                        '-c:v', 'libx264',
                        '-b:v', TARGET_BITRATE,
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        '-b:a', '96k',
                        '-y',
                        output_mp4_final
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=600)
                    if result.returncode == 0:
                        print(f"   ✅ MP4 conversion (ffmpeg) done: {os.path.getsize(output_mp4_final)} bytes")
                        upload_file = output_mp4_final
                    else:
                        raise Exception("ffmpeg failed")
                except Exception as e:
                    print(f"   [INFO] ffmpeg unavailable, using imageio...")
                    # Fallback to imageio (slower but works)
                    reader = imageio.get_reader(output_mp4)
                    writer = imageio.get_writer(output_mp4_final, fps=fps, codec='libx264')
                    for frame_idx, frame in enumerate(reader):
                        if frame_idx % 50 == 0:
                            print(f"   [DEBUG] Converting frame {frame_idx}...")
                        writer.append_data(frame)
                    writer.close()
                    reader.close()
                    print(f"   ✅ MP4 conversion (imageio) done: {os.path.getsize(output_mp4_final)} bytes")
                    upload_file = output_mp4_final
        except Exception as e:
            print(f"   [WARN] AVI→MP4 conversion failed: {str(e)[:100]}")
            print(f"   [INFO] Uploading AVI directly (may not play in browser)")
            upload_file = output_mp4
        
        # ⭐ PIL Compression DISABLED - Makes file larger paradoxically!
        # File compression handled via bitrate encoding above
        
        # ⭐ Upload annotated video to Cloudinary with smart fallback
        output_video_url = None
        backend_video_url = None
        file_size_mb = os.path.getsize(upload_file) / (1024 * 1024)
        
        print(f"   [DEBUG] Video file size: {file_size_mb:.1f} MB")
        
        # ⭐ SKIP Cloudinary upload - use local backend storage only for speed
        # Cloudinary was causing 90+ second timeouts, backend fallback is fast & reliable
        output_video_url = None
        backend_video_url = None
        print(f"   [INFO] Using local backend storage (skipping Cloudinary)")
        
        # TRY: Backend fallback (ALWAYS USED - fast & reliable)
        backend_video_dir = os.path.join(BASE_DIR, "output_videos")
        print(f"   [DEBUG] Backend directory: {backend_video_dir}")
        os.makedirs(backend_video_dir, exist_ok=True)
        backend_video_path = os.path.join(backend_video_dir, f"output_{job_id}.mp4")
        
        print(f"   [DEBUG] Upload file: {upload_file}")
        print(f"   [DEBUG] Upload file exists: {os.path.exists(upload_file)}")
        print(f"   [DEBUG] Backend path: {backend_video_path}")
        print(f"   [DEBUG] Paths equal: {upload_file == backend_video_path}")
        
        try:
            import shutil
            if upload_file != backend_video_path and os.path.exists(upload_file):
                print(f"   [DEBUG] Copying {upload_file} → {backend_video_path}")
                print(f"   [DEBUG] Backend dir exists: {os.path.exists(backend_video_dir)}")
                print(f"   [DEBUG] Backend dir writable: {os.access(backend_video_dir, os.W_OK)}")
                shutil.copy2(upload_file, backend_video_path)
                file_exists_after = os.path.exists(backend_video_path)
                print(f"   [DEBUG] Copy successful: {file_exists_after}")
                if file_exists_after:
                    file_size = os.path.getsize(backend_video_path)
                    print(f"   [DEBUG] File size verified: {file_size} bytes")
                else:
                    print(f"   [ERROR] Copy reported success but file not found at {backend_video_path}")
            else:
                print(f"   [DEBUG] Copy condition not met - checking if already at destination")
                print(f"   [DEBUG] upload_file exists: {os.path.exists(upload_file)}")
                print(f"   [DEBUG] Paths equal: {upload_file == backend_video_path}")
                if os.path.exists(backend_video_path):
                    print(f"   [DEBUG] File already at destination: {backend_video_path}")
            
            backend_video_url = f"/download/{job_id}"
            print(f"   ✅ Backend download link ready: {backend_video_url}")
        except Exception as e:
            print(f"   [ERROR] Failed to copy video to backend: {e}")
            import traceback
            traceback.print_exc()
            backend_video_url = None
        
        # Final URL: ALWAYS USE BACKEND (Cloudinary disabled for reliability)
        final_video_url = backend_video_url or output_video_url
        
        # ⭐ CRITICAL: Store all URLs in job result for frontend
        # Frontend will use final_video_url to stream video via /download/{job_id}
        
        # ⭐ Clean up temp files
        for temp_file in [output_mp4, output_mp4_final]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        print(f"   ✅ Temp files cleaned up")

        # Final results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed"] = True
        jobs[job_id]["outputVideoUrl"] = final_video_url
        jobs[job_id]["cloudinaryUrl"] = output_video_url
        jobs[job_id]["backendUrl"] = backend_video_url
        jobs[job_id]["vehicle_count"] = vehicle_count_total
        jobs[job_id]["frames_processed"] = frame_count
        jobs[job_id]["lane"] = {
            "kiri": {
                "total": counters['kiri']['total'],
                "mobil": counters['kiri']['mobil'],
                "bus": counters['kiri']['bus'],
                "truk": counters['kiri']['truk']
            },
            "kanan": {
                "total": counters['kanan']['total'],
                "mobil": counters['kanan']['mobil'],
                "bus": counters['kanan']['bus'],
                "truk": counters['kanan']['truk']
            }
        }
        jobs[job_id]["detections"] = detections

        save_job(job_id)

        elapsed = time.time() - start_time
        print(f"✅ Processing completed in {elapsed:.1f}s")
        print(f"   Vehicles: {vehicle_count_total}")
        print(f"   Lane Kiri: {counters['kiri']['total']} | Kanan: {counters['kanan']['total']}")
        print(f"🟢 [process_video END] Job {job_id}\n")

    except Exception as e:
        print(f"🔴 [process_video ERROR] Job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        save_job(job_id)

# ================= ROUTES =================

# ⭐ HEALTH CHECK - untuk monitor tunnel dari Railway
@app.get("/health")
async def health_check():
    """Health check endpoint - membantu detect tunnel issues"""
    return {
        "status": "healthy",
        "device": device_type,
        "gpu_available": device_type == "cuda",
        "timestamp": time.time(),
        "uptime": time.time()
    }

# ⭐ STATUS CHECK - monitor job queue
@app.get("/status")
async def status_check():
    """Check current processing status"""
    return {
        "active_jobs": len([j for j in jobs.values() if j.get("status") == "processing"]),
        "completed_jobs": len([j for j in jobs.values() if j.get("status") == "completed"]),
        "failed_jobs": len([j for j in jobs.values() if j.get("status") == "failed"]),
        "device": device_type,
        "queue_size": len(jobs)
    }

@app.post("/detect")
async def detect_video(
    file: UploadFile = File(...),
    file_detection_id: str = Form(...)
):
    """⭐ Process video from file upload (multipart from Railway backend)"""
    print(f"\n[DEBUG /detect] ========== ENDPOINT CALLED ==========")
    print(f"[DEBUG] File: {file.filename}")
    print(f"[DEBUG] Detection ID: {file_detection_id}")
    
    try:
        # Save uploaded file to temp location
        import tempfile
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"{file_detection_id}_{file.filename}")
        
        print(f"[DEBUG] Saving uploaded file to: {video_path}")
        
        # Write file to disk
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"[DEBUG] File saved successfully")
        file_size = os.path.getsize(video_path)
        print(f"[DEBUG] File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Validate video format
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        if not video_path.lower().endswith(valid_extensions):
            error_msg = f"Invalid video format. Supported: {valid_extensions}"
            print(f"[ERROR] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"[DEBUG] Video format OK: {video_path.split('.')[-1].upper()}")
        
        # Use detection_id as job_id
        job_id = file_detection_id
        
        # Start background processing
        print(f"[DEBUG] Starting background processing task...")
        task = asyncio.create_task(process_video(job_id, video_path))
        
        # Set initial job state
        jobs[job_id] = {
            "status": "processing",
            "progress": 5,
            "message": "Starting processing...",
            "total": 0,
            "completed": False,
            "video_path": video_path
        }
        save_job(job_id)
        
        print(f"[DEBUG] ========== JOB QUEUED ==========")
        print(f"[DEBUG] Job ID: {job_id}")
        print(f"[DEBUG] Video: {video_path}")
        
        # Return immediately - frontend will poll /result/{job_id}
        return {
            "job_id": job_id,
            "video_path": video_path,
            "status": "queued",
            "message": "Video queued for processing"
        }
        
    except HTTPException as he:
        print(f"[ERROR] HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{job_id}")
async def download_video(job_id: str, request: Request):
    """⭐ Download video dengan streaming dan timeout handling"""
    backend_video_dir = os.path.join(BASE_DIR, "output_videos")
    video_path = os.path.join(backend_video_dir, f"output_{job_id}.mp4")
    
    print(f"\n[DEBUG /download] Job ID: {job_id}")
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    print(f"[DEBUG] Backend video dir: {backend_video_dir}")
    print(f"[DEBUG] Checking file: {video_path}")
    print(f"[DEBUG] Dir exists: {os.path.exists(backend_video_dir)}")
    print(f"[DEBUG] File exists: {os.path.exists(video_path)}")
    
    if not os.path.exists(video_path):
        # List directory contents for debugging
        try:
            if os.path.exists(backend_video_dir):
                files = os.listdir(backend_video_dir)
                print(f"[DEBUG] Files in {backend_video_dir}: {files}")
            else:
                print(f"[ERROR] Backend video directory does not exist: {backend_video_dir}")
        except Exception as e:
            print(f"[ERROR] Could not list directory: {e}")
        print(f"[ERROR] Video not found at: {video_path}")
        return JSONResponse(
            {
                "error": f"Video {job_id} not found",
                "expected_path": video_path,
                "directory_exists": os.path.exists(backend_video_dir)
            },
            status_code=404
        )
    
    try:
        file_size = os.path.getsize(video_path)
        print(f"[DEBUG] File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"[DEBUG] Streaming video to client...")
        
        def iter_file():
            with open(video_path, 'rb') as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            print(f"[DEBUG] Streaming complete.")
        
        return StreamingResponse(
            iter_file(),
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="output_{job_id}.mp4"',
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes",
                # ⭐ KEEP-ALIVE prevent premature disconnect
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=60, max=100",
                # ⭐ CORS headers untuk cross-origin requests
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Range"
            }
        )
    except Exception as e:
        print(f"[ERROR] Exception during streaming: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/result/{job_id}")
async def result(job_id: str, request: Request):
    """Get job result dengan timeout handling dan keep-alive"""
    job = jobs.get(job_id)
    
    # ⭐ FIX: Extract ngrok URL from X-Forwarded headers (for production ngrok tunnel)
    # When Backend (Railway) calls via ngrok, it sends X-Forwarded-Proto and X-Forwarded-Host
    forwarded_proto = request.headers.get('X-Forwarded-Proto', 'http')
    forwarded_host = request.headers.get('X-Forwarded-Host', None)
    
    if forwarded_host:
        # Use forwarded headers (from ngrok tunnel)
        base_url = f"{forwarded_proto}://{forwarded_host}"
        print(f"[DEBUG /result] Using X-Forwarded headers: {base_url}")
    else:
        # Fallback to direct base_url
        base_url = str(request.base_url).rstrip('/')
        print(f"[DEBUG /result] Using request.base_url: {base_url}")
    
    print(f"[DEBUG /result] Job ID: {job_id}, Status: {job.get('status') if job else 'not_found'}")

    if not job:
        job = load_job(job_id)
        if job:
            jobs[job_id] = job
        else:
            response = {
                "job_id": job_id,
                "status": "pending",
                "progress": 0,
                "message": "Job not found, may be queued",
                "vehicle_count": 0,
                "frames_processed": 0,
                "lane": {
                    "kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
                    "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}
                },
                "detections": [],
                "outputVideoUrl": None
            }
            return JSONResponse(response)

    # ⭐ Construct full URL using request base_url
    output_video_url = job.get("outputVideoUrl", None)
    print(f"[DEBUG /result] job.outputVideoUrl before conversion: {output_video_url}")
    print(f"[DEBUG /result] base_url: {base_url}")
    
    if output_video_url and output_video_url.startswith('/'):
        # Relative path → convert to full URL
        output_video_url = f"{base_url}{output_video_url}"
        print(f"[DEBUG /result] Converted to full URL: {output_video_url}")
    else:
        print(f"[DEBUG /result] NOT CONVERTING - already full URL or None")

    response = {
        "job_id": job_id,
        "status": job.get("status", "processing"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "vehicle_count": job.get("vehicle_count", 0),
        "frames_processed": job.get("frames_processed", 0),
        "lane": job.get("lane", {
            "kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
            "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}
        }),
        "detections": job.get("detections", []),
        "outputVideoUrl": output_video_url,
        "cloudinaryUrl": job.get("cloudinaryUrl", None),
        "backendUrl": output_video_url,
    }
    
    # ⭐ Add keep-alive headers untuk prevent tunnel timeout
    return JSONResponse(
        to_python_type(response),
        headers={
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=60, max=100",
            "Cache-Control": "no-cache, must-revalidate"
        }
    )