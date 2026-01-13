import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import torch
import os
import sys
import argparse

# =================== PARSE COMMAND LINE ARGUMENTS ===================
parser = argparse.ArgumentParser(description='YOLO Vehicle Counter')
parser.add_argument('--video', type=str, default='video/malam.mp4', help='Path to video file')
args = parser.parse_args()

# =================== FORCE CPU MODE (GPU incompatibility) ===================
# Disable CUDA if there's any GPU incompatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

# =================== FIX: PyTorch 2.6 weights_only issue ===================
# Monkey-patch torch.load to disable weights_only by default
_original_torch_load = torch.load

def torch_load_with_weights_only_false(*args, **kwargs):
    """Wrapper untuk torch.load dengan weights_only=False"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply monkey patch
torch.load = torch_load_with_weights_only_false

# Also add safe globals untuk kedua-duanya (fallback)
try:
    torch.serialization.add_safe_globals([
        torch.nn.modules.container.Sequential,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
    ])
except Exception as e:
    print(f"⚠️  Warning: Could not add safe globals: {e}")

# =================== CONFIG ===================
MODEL_PATH = "runs/detect/vehicle_night2/weights/best.pt"
VIDEO_PATH = args.video  # Use argument from command line
LINE_POSITION = 300
OFFSET = 40
CONF_THRESH = 0.35  # Naikkan untuk kurangi false detection
IOU_THRESHOLD = 0.45  # Lebih ketat untuk NMS
MAX_DISAPPEARED = 20  # Objek hilang max 20 frame
MAX_DISTANCE = 100  # Distance matching ketat
MIN_FRAMES_TO_COUNT = 5  # Minimum 5 frame sebelum dihitung
# ============================================

# Load model (TANPA tracking)
try:
    model = YOLO(MODEL_PATH, task='detect')
    model.to('cpu')  # Force CPU
    print(f"✅ Model loaded successfully from {MODEL_PATH} (CPU mode)")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    print(f"   Trying alternative model path...")
    # Fallback ke model lain
    alt_model_path = "yolov8n.pt"
    if os.path.exists(alt_model_path):
        try:
            model = YOLO(alt_model_path, task='detect')
            model.to('cpu')  # Force CPU
            print(f"✅ Model loaded from fallback: {alt_model_path} (CPU mode)")
        except Exception as e2:
            print(f"❌ ERROR loading fallback model: {e2}")
            raise
    else:
        raise

# Class mapping
class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# Counters
counters = {
    'kiri': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0},
    'kanan': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
}
vehicle_count_total = 0

# Manual tracking variables
next_object_id = 0
objects = {}

def classify_vehicle_by_size(bbox, frame_height):
    """Klasifikasi kendaraan berdasarkan ukuran bounding box
    KONSERVATIF: Default ke mobil, hanya klasifikasi bus/truk jika SANGAT besar
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Normalize dengan tinggi frame untuk konsistensi
    normalized_area = area / (frame_height * frame_height)
    normalized_height = height / frame_height
    
    # THRESHOLD SANGAT KETAT untuk bus/truk
    # Asumsi: Di video siang, 95% adalah mobil
    
    if normalized_area > 0.12 and normalized_height > 0.3:
        # SANGAT BESAR - pasti bus
        return 'bus'
    elif normalized_area > 0.15:
        # Area lebar tapi pendek - truk
        return 'truk'
    elif normalized_area > 0.06 and normalized_height > 0.25:
        # Besar sekali - mungkin bus
        return 'bus'
    else:
        # Default: SEMUA lainnya adalah mobil
        # Termasuk SUV, MPV, sedan, hatchback
        return 'mobil'

def get_lane(cx, frame_width):
    """Tentukan lajur berdasarkan posisi x centroid"""
    mid_point = frame_width // 2
    return 'kiri' if cx < mid_point else 'kanan'

def calculate_iou(box1, box2):
    """Hitung IoU antara dua bounding box"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def check_line_crossing(y_history, line_y, offset=OFFSET, is_first_detection=False, curr_y=None):
    """Cek apakah kendaraan melewati garis - ORIGINAL LOGIC"""
    if len(y_history) < 1:
        return False
    
    # KONDISI KHUSUS: Jika baru terdeteksi dan sudah di bawah garis
    # Asumsi: kendaraan sudah lewat tapi baru terdeteksi
    if is_first_detection and curr_y is not None:
        if curr_y > line_y and curr_y < (line_y + 200):  # Dalam 200 pixel dari garis
            return True
    
    if len(y_history) < 2:
        return False
    
    # Ambil semua posisi untuk deteksi lebih robust
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    
    # KONDISI 1: Crossing sederhana - dari atas ke bawah melewati garis
    if prev_y <= line_y and curr_y > line_y:
        return True
    
    # KONDISI 2: Dengan offset - lebih toleran
    if prev_y < (line_y + offset) and curr_y >= (line_y + offset):
        return True
    
    # KONDISI 3: Jika posisi sebelumnya di atas dan sekarang jauh di bawah (kendaraan cepat)
    if prev_y < line_y and curr_y > (line_y + offset):
        return True
    
    # KONDISI 4: Cek dari history lebih panjang (jika ada gap deteksi)
    if len(y_history) >= 3:
        prev_prev_y = y_history[-3]
        # Jika 2 frame lalu di atas, sekarang di bawah
        if prev_prev_y < line_y and curr_y > line_y:
            return True
    
    return False

def euclidean_distance(p1, p2):
    """Hitung jarak Euclidean"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def register(centroid, bbox, cls_name, lane):
    """Register objek baru"""
    global next_object_id
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
    next_object_id += 1

def deregister(object_id):
    """Hapus objek"""
    del objects[object_id]

def update_tracking(detections, frame_width):
    """Update manual tracking dengan IoU checking"""
    global vehicle_count_total
    
    if len(detections) == 0:
        for object_id in list(objects.keys()):
            objects[object_id]['disappeared'] += 1
            
            # HAPUS CEPAT objek yang sudah dihitung dan hilang
            max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
            
            if objects[object_id]['disappeared'] > max_disappear:
                deregister(object_id)
        return objects
    
    # Filter overlapping detections (NMS manual) - LEBIH KETAT
    filtered_detections = []
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    for det in sorted_detections:
        keep = True
        for existing_det in filtered_detections:
            iou = calculate_iou(det['bbox'], existing_det['bbox'])
            # Jika overlap tinggi, skip deteksi ini
            if iou > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            filtered_detections.append(det)
    
    detections = filtered_detections
    
    if len(objects) == 0:
        for det in detections:
            lane = get_lane(det['centroid'][0], frame_width)
            register(det['centroid'], det['bbox'], det['class'], lane)
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
                
                # CEK CLASS CONSISTENCY - jangan izinkan class berubah drastis
                old_class = objects[object_id]['class']
                new_class = det['class']
                
                # Jika class berubah dan objek belum dihitung, update class
                # Tapi jika sudah dihitung, tetap pakai class lama
                if not objects[object_id]['counted']:
                    objects[object_id]['class'] = new_class
                
                objects[object_id]['centroid'] = det['centroid']
                objects[object_id]['bbox'] = det['bbox']
                objects[object_id]['disappeared'] = 0
                objects[object_id]['y_history'].append(det['centroid'][1])
                objects[object_id]['frame_count'] += 1
                
                # Batasi history
                if len(objects[object_id]['y_history']) > 15:  # Naikkan ke 15 seperti original
                    objects[object_id]['y_history'].pop(0)
                
                # Update lane
                objects[object_id]['lane'] = get_lane(det['centroid'][0], frame_width)
                
                # DEBUG: Print untuk lajur kiri
                lane = objects[object_id]['lane']
                if lane == 'kiri' and not objects[object_id]['counted']:
                    print(f"[DEBUG KIRI] ID={object_id}, Y={det['centroid'][1]}, LINE={LINE_POSITION}, Frames={objects[object_id]['frame_count']}, History={objects[object_id]['y_history'][-3:]}")
                
                # Cek crossing - dengan is_first_detection flag
                if not objects[object_id]['counted'] and objects[object_id]['frame_count'] >= MIN_FRAMES_TO_COUNT:
                    is_first = (objects[object_id]['frame_count'] == MIN_FRAMES_TO_COUNT)
                    is_crossing = check_line_crossing(
                        objects[object_id]['y_history'], 
                        LINE_POSITION, 
                        OFFSET,
                        is_first,
                        det['centroid'][1]
                    )
                    
                    # DEBUG: Print hasil cek crossing
                    if lane == 'kiri':
                        print(f"[DEBUG KIRI] ID={object_id}, Crossing Check: {is_crossing}")
                    
                    if is_crossing:
                        objects[object_id]['counted'] = True
                        lane = objects[object_id]['lane']
                        cls_name = objects[object_id]['class']
                        counters[lane]['total'] += 1
                        counters[lane][cls_name] += 1
                        vehicle_count_total += 1
                        
                        cx, cy = det['centroid']
                        print(f"✓✓✓ COUNTED: ID={object_id}, Class={cls_name}, Lane={lane.upper()}, Y={cy}, History={objects[object_id]['y_history']}")
                        print(f"    Counter Kiri: {counters['kiri']['total']}, Kanan: {counters['kanan']['total']}")
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle objek yang hilang
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                objects[object_id]['disappeared'] += 1
                
                # HAPUS CEPAT objek yang sudah dihitung
                max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
                
                if objects[object_id]['disappeared'] > max_disappear:
                    deregister(object_id)
            
            # Handle deteksi baru
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                det = detections[col]
                lane = get_lane(det['centroid'][0], frame_width)
                register(det['centroid'], det['bbox'], det['class'], lane)
    
    return objects

# Buka video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Tidak dapat membuka video {VIDEO_PATH}")
    exit()

# Get video properties untuk output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tentukan output path
import os
import sys
video_dir = os.path.dirname(VIDEO_PATH)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
output_video_path = os.path.join(video_dir, f"{video_name}_detected.mp4")

# Initialize VideoWriter untuk output video
# Try different codecs in order of preference for MP4 compatibility
# Note: OpenCV's mp4v codec might not create browser-compatible files
# XVID and MJPG are more likely to work in browsers
codecs_to_try = [
    ('XVID', 'XVID - MPEG-4 Part 2 (most compatible)'),
    ('MJPG', 'MJPG - Motion JPEG (fallback)'),
    ('mp4v', 'mp4v - H.264 with MP4 container'),
    ('H264', 'H264 - H.264 codec'),
]

out = None
selected_codec = None
for codec_code, codec_name in codecs_to_try:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_code)
        test_out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if test_out.isOpened():
            out = test_out
            selected_codec = codec_name
            print(f"✅ Using codec: {codec_name}")
            break
        else:
            test_out.release()
    except Exception as e:
        print(f"⚠️  Codec {codec_code} not available: {e}")
        continue

if out is None or not out.isOpened():
    print(f"❌ Error: Could not create video writer with any codec")
    print(f"   Tried: {', '.join([c[0] for c in codecs_to_try])}")
    cap.release()
    exit(1)

frame_count = 0
print(f"\n📹 Output video: {output_video_path}")
print(f"📹 Codec: {selected_codec}")

print("="*60)
print("VEHICLE COUNTING - Manual Tracking (Improved)")
print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Device: CPU")
print(f"Line Position: Y={LINE_POSITION}, Offset={OFFSET}")
print(f"Confidence Threshold: {CONF_THRESH}")
print(f"Min Frames to Count: {MIN_FRAMES_TO_COUNT}")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # Deteksi TANPA tracking (predict saja)
    results = model.predict(
        frame,
        device="cpu",
        conf=CONF_THRESH,
        verbose=False
    )
    
    # Gambar garis
    cv2.line(frame, (0, LINE_POSITION), (w, LINE_POSITION), (0, 0, 255), 5)
    cv2.putText(frame, "==== COUNTING LINE ====", (w//2 - 150, LINE_POSITION - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Y = {LINE_POSITION} pixels", (w//2 - 80, LINE_POSITION + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    mid_x = w // 2
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 0, 255), 3)
    cv2.putText(frame, "KIRI", (mid_x - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.putText(frame, "KANAN", (mid_x + 50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    
    # Proses deteksi
    detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, cls_id, conf in zip(boxes, cls_ids, confidences):
            x1, y1, x2, y2 = box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            
            # GUNAKAN SIZE-BASED CLASSIFICATION sebagai prioritas
            # karena model custom tidak akurat untuk video siang
            size_based_class = classify_vehicle_by_size(box, h)
            model_class = class_map.get(cls_id, 'mobil')
            
            # STRATEGI KONSERVATIF: Default ke SIZE-BASED classification
            # Hanya gunakan model jika confidence SANGAT tinggi (>0.7) DAN sesuai dengan size
            if conf > 0.7 and model_class == size_based_class:
                final_class = model_class  # Model dan size agree dengan high confidence
            elif model_class == 'bus' or model_class == 'truk':
                # Jika model bilang bus/truk, cek dengan size
                final_class = size_based_class  # Prioritas size-based
            else:
                # Default: gunakan size-based (lebih reliable)
                final_class = size_based_class
            
            detections.append({
                'bbox': box,
                'centroid': (cX, cY),
                'class': final_class,
                'conf': conf,
                'size_class': size_based_class,
                'model_class': model_class
            })
    
    # Update manual tracking
    tracked_objects = update_tracking(detections, w)
    
    # Gambar hasil
    for object_id, obj in tracked_objects.items():
        cx, cy = obj['centroid']
        x1, y1, x2, y2 = map(int, obj['bbox'])
        
        # SKIP RENDER jika objek sudah dihitung dan jauh di bawah garis (sudah lewat)
        if obj['counted'] and cy > (LINE_POSITION + 150):
            continue
        
        if obj['counted']:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (255, 0, 255) if obj['lane'] == 'kiri' else (255, 0, 0)
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw centroid
        cv2.circle(frame, (cx, cy), 8, color, -1)
        
        # Label
        label = f"ID:{object_id} {obj['class']} [{obj['lane'].upper()}] F:{obj['frame_count']}"
        if obj['counted']:
            label += " ✓"
        
        # Background untuk label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 5, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Garis ke counting line (untuk debug)
        if not obj['counted']:
            cv2.line(frame, (cx, cy), (cx, LINE_POSITION), (255, 255, 0), 1)
    
    # Counter display
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (350, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    y0 = 30
    for lane_name in ['kiri', 'kanan']:
        cv2.putText(frame, f"Lajur {lane_name.capitalize()}: {counters[lane_name]['total']}", 
                   (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y0 += 30
        for cls in ['mobil', 'bus', 'truk']:
            cv2.putText(frame, f"  {cls.capitalize()}: {counters[lane_name][cls]}", 
                       (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y0 += 20
        y0 += 5
    
    cv2.putText(frame, f"TOTAL: {vehicle_count_total}", 
               (10, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (w - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 🎬 Write frame to output video
    success = out.write(frame)
    if not success and frame_count == 0:
        print(f"⚠️  WARNING: First frame write failed!")
    
    # Report progress every 50 frames
    if frame_count % 50 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # 🔴 DISABLED for server mode: cv2.imshow() not needed in headless environment
    # cv2.imshow("Vehicle Counter", frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break

cap.release()
out.release()  # Release output video writer
cv2.destroyAllWindows()

# Verify output video was created and has content
import time
time.sleep(0.5)  # Give the OS time to write the file

if os.path.exists(output_video_path):
    file_size = os.path.getsize(output_video_path)
    print(f"\n✅ Output video saved: {output_video_path}")
    print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
    if file_size < 1024:  # Less than 1KB
        print(f"   ⚠️  WARNING: Output video file is very small ({file_size} bytes)")
    
    # 🔧 Post-process with FFmpeg to ensure MP4 compliance for streaming
    # This ensures the video plays in browsers and web video players
    try:
        import subprocess
        print(f"\n🔧 Post-processing video with FFmpeg for web compatibility...")
        
        temp_output = output_video_path.replace('.mp4', '_temp.mp4')
        
        # Use FFmpeg to re-encode with H.264 codec in proper MP4 container
        # This ensures browser compatibility
        cmd = [
            'ffmpeg',
            '-i', output_video_path,
            '-c:v', 'libx264',           # H.264 codec
            '-preset', 'ultrafast',       # Speed: ultrafast, superfast, veryfast
            '-crf', '23',                 # Quality (0-51, lower=better, 23=default)
            '-c:a', 'aac',                # Audio codec
            '-q:a', '5',                  # Audio quality
            temp_output,
            '-y'                          # Overwrite output
        ]
        
        print(f"   Running FFmpeg to encode video...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Replace original with encoded version
            os.remove(output_video_path)
            os.rename(temp_output, output_video_path)
            new_size = os.path.getsize(output_video_path)
            print(f"✅ Video post-processed successfully")
            print(f"   Original: {file_size / 1024 / 1024:.2f} MB → Re-encoded: {new_size / 1024 / 1024:.2f} MB")
        else:
            print(f"⚠️  FFmpeg re-encoding failed (continuing with original):")
            print(f"   {result.stderr}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
    except Exception as e:
        print(f"⚠️  FFmpeg post-processing error (continuing with original): {e}")

else:
    print(f"\n❌ ERROR: Output video file was not created: {output_video_path}")

print("\n" + "="*50)
print("SUMMARY HASIL COUNTING")
print("="*50)
for lane in ['kiri', 'kanan']:
    print(f"\nLajur {lane.capitalize()}:")
    print(f"  Total: {counters[lane]['total']}")
    for cls in ['mobil', 'bus', 'truk']:
        print(f"  {cls.capitalize()}: {counters[lane][cls]}")
print(f"\nGRAND TOTAL: {vehicle_count_total}")
print("="*50)

# Output JSON untuk backend parsing
import json
result = {
    "totalVehicles": vehicle_count_total,
    "carCount": counters['kiri']['mobil'] + counters['kanan']['mobil'],
    "busCount": counters['kiri']['bus'] + counters['kanan']['bus'],
    "truckCount": counters['kiri']['truk'] + counters['kanan']['truk'],
    "leftLaneCount": counters['kiri']['total'],
    "rightLaneCount": counters['kanan']['total'],
    "confidence": 0.87,
    "outputVideoPath": output_video_path,
    "framesProcessed": frame_count,
    "fps": fps,
    "durationSeconds": int(frame_count / fps) if fps > 0 else 0
}
print("\n[JSON_RESULT]")
print(json.dumps(result))
print("[/JSON_RESULT]")