# api.py - YOLO Vehicle Detection with Manual Tracking
import os
import asyncio
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

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

# ================= MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

jobs = {}

# ================= TRACKING CONFIG =================
LINE_POSITION = 300
OFFSET = 40
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.4
MAX_DISAPPEARED = 15
MAX_DISTANCE = 120
MIN_FRAMES_TO_COUNT = 2

# Class mapping
class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# ================= TRACKING FUNCTIONS =================

def get_lane(cx, frame_width):
    """Tentukan lajur berdasarkan posisi x centroid"""
    mid_point = frame_width // 2
    return 'kiri' if cx < mid_point else 'kanan'

def calculate_iou(box1, box2):
    """Hitung IoU antara dua bounding box"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def check_line_crossing(y_history, line_y, offset=OFFSET, is_first_detection=False, curr_y=None):
    """Cek apakah kendaraan melewati garis"""
    if len(y_history) < 1:
        return False
    
    # KONDISI KHUSUS: Jika baru terdeteksi dan sudah di bawah garis
    if is_first_detection and curr_y is not None:
        if curr_y > line_y and curr_y < (line_y + 200):
            return True
    
    if len(y_history) < 2:
        return False
    
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    
    # KONDISI 1: Crossing sederhana
    if prev_y <= line_y and curr_y > line_y:
        return True
    
    # KONDISI 2: Dengan offset
    if prev_y < (line_y + offset) and curr_y >= (line_y + offset):
        return True
    
    # KONDISI 3: Cepat crossing
    if prev_y < line_y and curr_y > (line_y + offset):
        return True
    
    # KONDISI 4: History lebih panjang
    if len(y_history) >= 3:
        prev_prev_y = y_history[-3]
        if prev_prev_y < line_y and curr_y > line_y:
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

def update_tracking(objects, next_object_id, detections, frame_width, counters, vehicle_count_total):
    """Update manual tracking dengan IoU checking"""
    
    if len(detections) == 0:
        for object_id in list(objects.keys()):
            objects[object_id]['disappeared'] += 1
            max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
            
            if objects[object_id]['disappeared'] > max_disappear:
                deregister(objects, object_id)
        return objects, next_object_id, vehicle_count_total
    
    # Filter overlapping detections
    filtered_detections = []
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    for det in sorted_detections:
        keep = True
        for existing_det in filtered_detections:
            iou = calculate_iou(det['bbox'], existing_det['bbox'])
            if iou > IOU_THRESHOLD:
                keep = False
                break
        if keep:
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
        
        # Hitung jarak
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
                
                if distances[row, col] > MAX_DISTANCE:
                    continue
                
                object_id = object_ids[row]
                det = detections[col]
                
                if not objects[object_id]['counted']:
                    objects[object_id]['class'] = det['class']
                
                objects[object_id]['centroid'] = det['centroid']
                objects[object_id]['bbox'] = det['bbox']
                objects[object_id]['disappeared'] = 0
                objects[object_id]['y_history'].append(det['centroid'][1])
                objects[object_id]['frame_count'] += 1
                
                if len(objects[object_id]['y_history']) > 15:
                    objects[object_id]['y_history'].pop(0)
                
                objects[object_id]['lane'] = get_lane(det['centroid'][0], frame_width)
                
                # Cek crossing
                if not objects[object_id]['counted'] and objects[object_id]['frame_count'] >= MIN_FRAMES_TO_COUNT:
                    is_first = (objects[object_id]['frame_count'] == MIN_FRAMES_TO_COUNT)
                    is_crossing = check_line_crossing(
                        objects[object_id]['y_history'],
                        LINE_POSITION,
                        OFFSET,
                        is_first,
                        det['centroid'][1]
                    )
                    
                    if is_crossing:
                        objects[object_id]['counted'] = True
                        lane = objects[object_id]['lane']
                        cls_name = objects[object_id]['class']
                        counters[lane]['total'] += 1
                        counters[lane][cls_name] += 1
                        vehicle_count_total += 1
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle objek hilang
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                objects[object_id]['disappeared'] += 1
                max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
                
                if objects[object_id]['disappeared'] > max_disappear:
                    deregister(objects, object_id)
            
            # Handle deteksi baru
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                det = detections[col]
                lane = get_lane(det['centroid'][0], frame_width)
                next_object_id = register(objects, next_object_id, det['centroid'], det['bbox'], det['class'], lane)
    
    return objects, next_object_id, vehicle_count_total

# ================= HELPERS =================
async def upload_to_cloudinary(file: UploadFile):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = cloudinary.uploader.upload(
        tmp_path,
        resource_type="video",
        folder="input_videos"
    )
    os.remove(tmp_path)
    return result["secure_url"]

# ================= PROCESS =================
async def process_video(job_id: str, video_url: str):
    """Process video with manual tracking untuk accurate vehicle counting"""
    
    # Initialize tracking
    objects = {}
    next_object_id = 0
    counters = {
        'kiri': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0},
        'kanan': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
    }
    vehicle_count_total = 0
    
    jobs[job_id] = {
        "progress": 0,
        "total": 0,
        "kiri": 0,
        "kanan": 0,
        "detections": []
    }
    
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        jobs[job_id]["progress"] = -1
        jobs[job_id]["error"] = "Cannot open video"
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = 1
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w, _ = frame.shape
        
        # Predict
        results = model.predict(
            frame,
            device="cpu",
            conf=CONF_THRESH,
            verbose=False
        )
        
        # Process detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, cls_ids, confidences):
                x1, y1, x2, y2 = box
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'centroid': (cX, cY),
                    'class': class_map.get(cls_id, 'mobil'),
                    'conf': conf
                })
        
        # Update tracking
        objects, next_object_id, vehicle_count_total = update_tracking(
            objects, next_object_id, detections, w, counters, vehicle_count_total
        )
        
        # Update progress
        jobs[job_id]["progress"] = int((frame_count / total_frames) * 100)
        jobs[job_id]["total"] = vehicle_count_total
        jobs[job_id]["kiri"] = counters['kiri']['total']
        jobs[job_id]["kanan"] = counters['kanan']['total']
        
        await asyncio.sleep(0)
    
    cap.release()
    
    # Final results - NORMALIZED RESPONSE
    jobs[job_id]["progress"] = 100
    jobs[job_id]["completed"] = True
    
    # 🔴 CRITICAL: Explicit field names to prevent mapping confusion
    jobs[job_id]["vehicle_count"] = vehicle_count_total  # Unique vehicles counted
    jobs[job_id]["frames_processed"] = frame_count  # Total frames processed
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
    
    # DEBUG: Print detailed class distribution
    total_mobil = counters['kiri']['mobil'] + counters['kanan']['mobil']
    total_bus = counters['kiri']['bus'] + counters['kanan']['bus']
    total_truk = counters['kiri']['truk'] + counters['kanan']['truk']
    print(f"📊 CLASS DISTRIBUTION:")
    print(f"   Mobil: {total_mobil} ({total_mobil/vehicle_count_total*100:.1f}%)")
    print(f"   Bus: {total_bus} ({total_bus/vehicle_count_total*100:.1f}%)")
    print(f"   Truk: {total_truk} ({total_truk/vehicle_count_total*100:.1f}%)")
    
    # Validation: Ensure vehicle_count !== frames_processed
    if jobs[job_id]["vehicle_count"] == jobs[job_id]["frames_processed"]:
        print(f"⚠️ WARNING: vehicle_count ({jobs[job_id]['vehicle_count']}) == frames_processed ({jobs[job_id]['frames_processed']})")
        print("   This indicates a counting error!")
    
    print(f"✅ Final: {jobs[job_id]['vehicle_count']} vehicles in {jobs[job_id]['frames_processed']} frames")

# ================= ROUTES =================
@app.post("/detect")
async def detect_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    video_url = await upload_to_cloudinary(file)
    job_id = str(int(asyncio.get_event_loop().time() * 1000))
    background_tasks.add_task(process_video, job_id, video_url)
    return {"job_id": job_id, "video_url": video_url}

@app.get("/progress/{job_id}")
async def progress(job_id: str):
    async def stream():
        while True:
            job = jobs.get(job_id)
            if not job:
                break
            yield f"data: {job['progress']}\n\n"
            if job["progress"] >= 100:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/result/{job_id}")
async def result(job_id: str):
    """
    Return normalized response with explicit field names
    to prevent vehicle_count being confused with frames
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    
    # Validate response before returning
    if job.get("completed"):
        # Ensure vehicle_count is present and different from frames
        vehicle_count = job.get("vehicle_count", 0)
        frames = job.get("frames_processed", 0)
        
        if vehicle_count == frames and frames > 100:
            print(f"🔴 ERROR: vehicle_count == frames_processed. This is a mapping error!")
            print(f"   vehicle_count: {vehicle_count}")
            print(f"   frames_processed: {frames}")
            # Still return, but add warning flag
            job["_mapping_error"] = True
    
    return JSONResponse({
        "job_id": job_id,
        "status": "completed" if job.get("completed") else "processing",
        "progress": job.get("progress", 0),
        
        # 🔴 CRITICAL: Explicit vehicle counting (NOT frames)
        "vehicle_count": job.get("vehicle_count", 0),  # Unique vehicles
        "frames_processed": job.get("frames_processed", 0),  # Total frames
        
        # Lane breakdown
        "lane": job.get("lane", {
            "kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
            "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}
        }),
        
        # Safety flag
        "_mapping_error": job.get("_mapping_error", False),
        "_warning": "If vehicle_count == frames_processed, there's a counting error"
    })
