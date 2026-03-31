#!/usr/bin/env python3
"""
Vehicle Counter with Manual Tracking
Using line crossing detection for accurate vehicle counting
"""

import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os
import sys
import argparse
import json
import gc

# =================== FORCE CPU MODE ===================
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

_original_torch_load = torch.load
def torch_load_with_weights_only_false(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_with_weights_only_false

# =================== PARSE ARGUMENTS ===================
parser = argparse.ArgumentParser(description='YOLO Vehicle Counter')
parser.add_argument('--video', type=str, required=True, help='Path to video file')
args = parser.parse_args()

# =================== CONFIG ===================
MODEL_PATH = "runs/detect/vehicle_night2/weights/best.pt"
VIDEO_PATH = args.video
LINE_POSITION = 300
OFFSET = 40
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.4
MAX_DISAPPEARED = 15
MAX_DISTANCE = 120
MIN_FRAMES_TO_COUNT = 2

# Class mapping
class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# Load model
try:
    model = YOLO(MODEL_PATH, task='detect')
    model.to('cpu')
    print(f"✅ Model loaded: {MODEL_PATH} (CPU mode)", file=sys.stderr)
except Exception as e:
    print(f"❌ Model loading failed: {e}", file=sys.stderr)
    alt_model_path = "yolov8n.pt"
    if os.path.exists(alt_model_path):
        try:
            model = YOLO(alt_model_path, task='detect')
            model.to('cpu')
            print(f"✅ Using fallback model: {alt_model_path}", file=sys.stderr)
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(1)

class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# =================== TRACKING FUNCTIONS ===================

def classify_vehicle_by_size(bbox, frame_height):
    """Classify vehicle by bounding box size"""
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
    """Determine lane based on x position"""
    mid_point = frame_width // 2
    return 'kiri' if cx < mid_point else 'kanan'

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
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
    """Check if vehicle crosses the counting line"""
    if len(y_history) < 1:
        return False
    
    if is_first_detection and curr_y is not None:
        if curr_y > line_y and curr_y < (line_y + 200):
            return True
    
    if len(y_history) < 2:
        return False
    
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    
    # KONDISI 1: Simple crossing
    if prev_y <= line_y and curr_y > line_y:
        return True
    
    # KONDISI 2: With offset
    if prev_y < (line_y + offset) and curr_y >= (line_y + offset):
        return True
    
    # KONDISI 3: Fast crossing
    if prev_y < line_y and curr_y > (line_y + offset):
        return True
    
    # KONDISI 4: Long history
    if len(y_history) >= 3:
        prev_prev_y = y_history[-3]
        if prev_prev_y < line_y and curr_y > line_y:
            return True
    
    return False

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def register(objects, next_object_id, centroid, bbox, cls_name, lane):
    """Register new object"""
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
    """Remove object"""
    if object_id in objects:
        del objects[object_id]

def update_tracking(objects, next_object_id, detections, frame_width, counters, vehicle_count_total):
    """Update tracking with IoU checking"""
    
    if len(detections) == 0:
        for object_id in list(objects.keys()):
            objects[object_id]['disappeared'] += 1
            max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
            if objects[object_id]['disappeared'] > max_disappear:
                deregister(objects, object_id)
        return objects, next_object_id, vehicle_count_total
    
    # NMS: Filter overlapping detections
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
                
                # Check line crossing
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
                        
                        cx, cy = det['centroid']
                        print(f"✓ COUNTED: ID={object_id}, Class={cls_name}, Lane={lane.upper()}, Y={cy}", file=sys.stderr)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle missing objects
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                objects[object_id]['disappeared'] += 1
                max_disappear = 5 if objects[object_id]['counted'] else MAX_DISAPPEARED
                if objects[object_id]['disappeared'] > max_disappear:
                    deregister(objects, object_id)
            
            # Handle new detections
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                det = detections[col]
                lane = get_lane(det['centroid'][0], frame_width)
                next_object_id = register(objects, next_object_id, det['centroid'], det['bbox'], det['class'], lane)
    
    return objects, next_object_id, vehicle_count_total

# =================== MAIN PROCESSING ===================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}", file=sys.stderr)
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    total_frames = 1

print(f"[PROCESSING] Video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames", file=sys.stderr)

# Initialize counters
objects = {}
next_object_id = 0
counters = {
    'kiri': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0},
    'kanan': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
}
vehicle_count_total = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # YOLO detection
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
            
            # Size-based classification
            size_based_class = classify_vehicle_by_size(box, h)
            model_class = class_map.get(cls_id, 'mobil')
            
            if conf > 0.7 and model_class == size_based_class:
                final_class = model_class
            elif model_class == 'bus' or model_class == 'truk':
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
        objects, next_object_id, detections, w, counters, vehicle_count_total
    )
    
    if frame_count % 50 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"[PROGRESS] {progress:.1f}% ({frame_count}/{total_frames})", file=sys.stderr)

cap.release()

print(f"\n[SUMMARY] Processing complete", file=sys.stderr)
print("="*50, file=sys.stderr)
for lane in ['kiri', 'kanan']:
    print(f"Lajur {lane.capitalize()}: {counters[lane]['total']}", file=sys.stderr)
    for cls in ['mobil', 'bus', 'truk']:
        print(f"  {cls.capitalize()}: {counters[lane][cls]}", file=sys.stderr)
print(f"GRAND TOTAL: {vehicle_count_total}", file=sys.stderr)
print("="*50, file=sys.stderr)

# Output JSON result
result = {
    "totalVehicles": vehicle_count_total,
    "carCount": counters['kiri']['mobil'] + counters['kanan']['mobil'],
    "busCount": counters['kiri']['bus'] + counters['kanan']['bus'],
    "truckCount": counters['kiri']['truk'] + counters['kanan']['truk'],
    "leftLaneCount": counters['kiri']['total'],
    "rightLaneCount": counters['kanan']['total'],
    "confidence": 0.87,
    "framesProcessed": frame_count,
    "fps": fps,
    "durationSeconds": int(frame_count / fps) if fps > 0 else 0,
    "lane": {
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
}

# Output valid JSON
print(json.dumps(result))
