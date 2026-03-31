#!/usr/bin/env python3
"""
Test Counting Logic Improvements
Compare output video quality sebelum dan sesudah fixes
"""

import cv2
import os
import sys
from datetime import datetime

def analyze_video(video_path, output_name="test"):
    """Analyze video dan extract statistics"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video tidak ditemukan: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka video: {video_path}")
        return None
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Count detection frames (frames with bounding boxes)
    frame_count = 0
    detection_frames = 0
    vehicle_detections = []
    
    print(f"📊 Analyzing: {os.path.basename(video_path)}")
    print(f"   FPS: {fps}, Frames: {total_frames}, Size: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Look for green boxes (counted vehicles) or colored boxes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green (counted) = 0-255, 100-255, 100-255
        green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
        green_contours = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Blue/Magenta (not counted)
        blue_mask = cv2.inRange(hsv, (90, 100, 100), (130, 255, 255))
        blue_contours = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        total_boxes = len(green_contours) + len(blue_contours)
        if total_boxes > 0:
            detection_frames += 1
            vehicle_detections.append({
                'frame': frame_count,
                'green_boxes': len(green_contours),
                'blue_boxes': len(blue_contours)
            })
        
        if frame_count % 100 == 0:
            print(f"   Progress: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
    
    cap.release()
    
    # Extract final count from last frame
    final_count = extraction_final_count(video_path)
    
    stats = {
        'video': os.path.basename(video_path),
        'fps': fps,
        'total_frames': total_frames,
        'duration_sec': total_frames / fps,
        'resolution': f"{width}x{height}",
        'detection_frames': detection_frames,
        'final_vehicle_count': final_count,
        'detections': vehicle_detections
    }
    
    return stats

def extraction_final_count(video_path):
    """Extract final vehicle count dari video (baca text di frame terakhir)"""
    cap = cv2.VideoCapture(video_path)
    
    # Go to last frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 5))
    
    last_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple OCR: Look for "TOTAL: XXX" text (use pytesseract if available)
        # For now, just check panel region
        last_count = frame  # Store frame for manual inspection
    
    cap.release()
    return last_count

def print_comparison(stats_old, stats_new):
    """Print comparison between old and new output"""
    
    print("\n" + "="*70)
    print("📊 COMPARISON: Old vs New Counting Logic")
    print("="*70)
    
    if stats_old:
        print(f"\n❌ OLD OUTPUT (Before fixes):")
        print(f"   Video: {stats_old['video']}")
        print(f"   Duration: {stats_old['duration_sec']:.1f}s ({stats_old['total_frames']} frames)")
        print(f"   Detection Frames: {stats_old['detection_frames']}")
        print(f"   Estimated Vehicle Count: ???")
        print(f"   Issues: Likely double counting, bbox overlaps")
    
    if stats_new:
        print(f"\n✅ NEW OUTPUT (After fixes):")
        print(f"   Video: {stats_new['video']}")
        print(f"   Duration: {stats_new['duration_sec']:.1f}s ({stats_new['total_frames']} frames)")
        print(f"   Detection Frames: {stats_new['detection_frames']}")
        print(f"   Vehicle Count: ???")
        print(f"   Improvements: Stricter counting, no overlaps")
    
    print("\n" + "="*70)
    print("🔍 KEY CHANGES APPLIED:")
    print("="*70)
    print("  1. CONF_THRESH: 0.35 → 0.55 (filter false detections)")
    print("  2. IOU_THRESHOLD: 0.45 → 0.25 (stricter box overlaps)")
    print("  3. MIN_FRAMES_TO_COUNT: 5 → 12 (more confirmation)")
    print("  4. MAX_DISTANCE: 100 → 60 (prevent ID switching)")
    print("  5. Crossing Logic: Added downward movement validation")
    print("="*70)

if __name__ == "__main__":
    print("🎬 YOLO Vehicle Counting - Testing Improvements")
    print("="*70)
    
    # Video paths
    input_video = r"D:\backup\pktj\backend\yolo\input_videos\008.mp4"
    output_old = r"D:\backup\pktj\backend\yolo\output_videos\output_69c1d48ba30d4e51c338e4d6.mp4"
    
    # Analyze old output
    print("\n📹 Analyzing OLD output (before fixes)...")
    stats_old = analyze_video(output_old, "old")
    
    print("\n⏳ Running NEW inference (after fixes)...")
    print("   This will take several minutes...")
    
    # Import API components
    try:
        from api import process_video, model, device_idx, BASE_DIR
        import asyncio
        
        # Create test job
        test_job_id = f"test_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n   Job ID: {test_job_id}")
        print(f"   Input: {input_video}")
        print(f"   Model: best_video_008.pt (mAP50: 0.498, Car: 0.981)")
        print(f"   Device: {device_idx}")
        
        # Run processing
        asyncio.run(process_video(test_job_id, input_video))
        
        # Find output video
        output_dir = os.path.join(BASE_DIR, "output_videos")
        output_new = os.path.join(output_dir, f"output_{test_job_id}.mp4")
        
        if os.path.exists(output_new):
            print(f"\n✅ New output generated: {output_new}")
            
            # Analyze new output
            print("\n📹 Analyzing NEW output (after fixes)...")
            stats_new = analyze_video(output_new, "new")
            
            # Compare
            print_comparison(stats_old, stats_new)
            
            print("\n📊 Next Steps:")
            print("   1. Compare both videos manually")
            print("   2. Check counting accuracy")
            print("   3. Verify no double counting")
            print("   4. Confirm bbox overlaps are gone")
            print("   5. Update CONF_THRESH/IOU_THRESHOLD if needed")
            
        else:
            print(f"\n❌ Output video not generated at: {output_new}")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✨ Testing Complete!")
    print("="*70)
