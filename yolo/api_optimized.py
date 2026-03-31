# api.py - YOLO Vehicle Detection dengan Optimasi (Berbasis count_video.py)
import os
import asyncio
import tempfile
import cv2
import gc
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import time
try:
    import imageio
except ImportError:
    imageio = None

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
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

os.makedirs(JOBS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# In-memory jobs cache
jobs = {}

def save_job(job_id: str):
    """Save job to persistent file storage"""
    try:
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(job_file, 'w') as f:
            import json
            json.dump(jobs[job_id], f)
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
LINE_POSITION = 300
OFFSET = 40
CONF_THRESH = 0.35  # Naikkan untuk kurangi false detection
IOU_THRESHOLD = 0.45  # Lebih ketat untuk NMS
MAX_DISAPPEARED = 20
MAX_DISTANCE = 100
MIN_FRAMES_TO_COUNT = 5  # ⭐ Naikkan untuk accuracy

# Class mapping
class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

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

def check_line_crossing(y_history, line_y, offset=OFFSET, is_first_detection=False, curr_y=None):
    """Cek apakah kendaraan melewati garis (dari count_video.py - lebih accurate)"""
    if len(y_history) < 1:
        return False
    
    # Kondisi khusus: jika baru terdeteksi dan sudah di bawah garis
    if is_first_detection and curr_y is not None:
        if curr_y > line_y and curr_y < (line_y + 200):
            return True
    
    if len(y_history) < 2:
        return False
    
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    
    # Kondisi 1: Crossing sederhana
    if prev_y <= line_y and curr_y > line_y:
        return True
    
    # Kondisi 2: Dengan offset
    if prev_y < (line_y + offset) and curr_y >= (line_y + offset):
        return True
    
    # Kondisi 3: Cepat crossing
    if prev_y < line_y and curr_y > (line_y + offset):
        return True
    
    # Kondisi 4: History lebih panjang
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

def update_tracking(objects, next_object_id, detections, frame_width, frame_height, counters, vehicle_count_total):
    """Update tracking dengan logika dari count_video.py"""
    
    if len(detections) == 0:
        for object_id in list(objects.keys()):
            objects[object_id]['disappeared'] += 1
            max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
            if objects[object_id]['disappeared'] > max_disappear:
                deregister(objects, object_id)
        return objects, next_object_id, vehicle_count_total
    
    # Filter overlapping detections (NMS manual)
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
            "detections": []
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        
        print(f"   [DEBUG] Video: FPS={fps}, Frames={total_frames}")

        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(tempfile.gettempdir(), f"output_{job_id}.mp4")
        out_writer = None

        # ⭐ OPTIMASI: Predict langsung tanpa resize (lebih cepat!)
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"   [DEBUG] End of video at frame {frame_count}")
                break

            if frame is None or frame.size == 0:
                frame_count += 1
                continue

            frame_count += 1
            h, w, _ = frame.shape

            # ⭐ KEY OPTIMIZATION: Predict langsung pada frame original size
            # Tidak perlu resize untuk inference!
            try:
                results = model.predict(
                    frame,
                    device="cpu",
                    conf=CONF_THRESH,
                    verbose=False,
                    max_det=50
                )
            except Exception as e:
                print(f"   [WARN] Predict error at frame {frame_count}: {e}")
                continue

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

                    # ⭐ CLASSIFICATION LOGIC dari count_video.py
                    size_based_class = classify_vehicle_by_size(box, h)
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
                objects, next_object_id, detections, w, h, counters, vehicle_count_total
            )

            # Initialize writer after getting frame size
            if out_writer is None:
                out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

            if out_writer is not None:
                out_writer.write(frame)

            # Update progress
            current_progress = int((frame_count / total_frames) * 100)
            jobs[job_id]["progress"] = current_progress
            jobs[job_id]["total"] = vehicle_count_total
            jobs[job_id]["kiri"] = counters['kiri']['total']
            jobs[job_id]["kanan"] = counters['kanan']['total']

            # Save periodically
            if current_progress % 10 == 0:
                save_job(job_id)
                gc.collect()

            await asyncio.sleep(0)

        cap.release()
        if out_writer is not None:
            out_writer.release()

        # ⭐ Convert to MP4 using imageio (no ffmpeg needed)
        output_video_url = None
        upload_path = output_path
        mp4_path = output_path.replace(".avi", ".mp4")
        
        try:
            print(f"   [DEBUG] Converting to MP4...")
            conversion_success = False
            
            if imageio is not None:
                try:
                    print(f"   [DEBUG] Using imageio for conversion...")
                    reader = imageio.get_reader(output_path)
                    metadata = reader.get_meta_data()
                    video_fps = metadata.get('fps', 30)
                    
                    writer = imageio.get_writer(mp4_path, fps=video_fps, codec='libx264', pixelformat='yuv420p')
                    for frame in reader:
                        writer.append_data(frame)
                    writer.close()
                    reader.close()
                    
                    if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000:
                        upload_path = mp4_path
                        conversion_success = True
                        print(f"   ✅ Converted to MP4: {os.path.getsize(mp4_path)} bytes")
                except Exception as e:
                    print(f"   [WARN] imageio conversion failed: {str(e)[:100]}")
            
            if not conversion_success:
                print(f"   [DEBUG] Trying OpenCV conversion...")
                try:
                    avi_cap = cv2.VideoCapture(output_path)
                    if avi_cap.isOpened():
                        avi_fps = int(avi_cap.get(cv2.CAP_PROP_FPS))
                        if avi_fps <= 0:
                            avi_fps = 30
                        avi_width = int(avi_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        avi_height = int(avi_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
                        mp4_writer = cv2.VideoWriter(mp4_path, fourcc_mp4, avi_fps, (avi_width, avi_height))
                        
                        if mp4_writer.isOpened():
                            frame_count_convert = 0
                            while True:
                                ret, frame = avi_cap.read()
                                if not ret:
                                    break
                                mp4_writer.write(frame)
                                frame_count_convert += 1
                            
                            avi_cap.release()
                            mp4_writer.release()
                            
                            if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000:
                                upload_path = mp4_path
                                conversion_success = True
                                print(f"   ✅ Converted to MP4 using mp4v: {os.path.getsize(mp4_path)} bytes")
                        else:
                            avi_cap.release()
                except Exception as cv2_err:
                    print(f"   [WARN] OpenCV conversion failed: {str(cv2_err)[:100]}")
        except Exception as e:
            print(f"   [WARN] Conversion error: {e}")
        
        # Upload video
        try:
            result = cloudinary.uploader.upload(
                upload_path,
                resource_type="video",
                folder="output_videos",
                timeout=600
            )
            output_video_url = result["secure_url"]
            print(f"   ✅ Video uploaded: {output_video_url}")
        except Exception as e:
            print(f"   [WARN] Upload failed: {e}")

        # Clean up
        for path in [output_path, mp4_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

        # Final results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed"] = True
        jobs[job_id]["outputVideoUrl"] = output_video_url
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
@app.post("/detect")
async def detect_video(file: UploadFile = File(...)):
    """Upload dan process video"""
    print(f"[DEBUG] /detect endpoint called")
    upload_result = await upload_to_cloudinary(file)
    video_path = upload_result["tmp_path"]
    video_url = upload_result["cloudinary_url"]
    print(f"[DEBUG] Video saved to: {video_path}")
    
    job_id = str(int(time.time() * 1000))
    print(f"[DEBUG] Created job_id: {job_id}")
    
    try:
        asyncio.create_task(process_video(job_id, video_path))
        print(f"[DEBUG] Started background task")
    except Exception as e:
        print(f"[ERROR] Failed to create task: {e}")
        return {"job_id": job_id, "video_url": video_url, "error": str(e)}
    
    return {"job_id": job_id, "video_url": video_url}

@app.get("/result/{job_id}")
async def result(job_id: str):
    """Get job result"""
    job = jobs.get(job_id)

    if not job:
        job = load_job(job_id)
        if job:
            jobs[job_id] = job
        else:
            response = {
                "job_id": job_id,
                "status": "pending",
                "progress": 0,
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

    response = {
        "job_id": job_id,
        "status": "completed" if job.get("completed") else "processing",
        "progress": job.get("progress", 0),
        "vehicle_count": job.get("vehicle_count", 0),
        "frames_processed": job.get("frames_processed", 0),
        "lane": job.get("lane", {
            "kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
            "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}
        }),
        "detections": job.get("detections", []),
        "outputVideoUrl": job.get("outputVideoUrl", None)
    }
    
    return JSONResponse(to_python_type(response))
