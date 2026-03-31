#!/usr/bin/env python3
"""
Diagnostic Tool: Analyze Line Position & Crossing Detection
Untuk debug kenapa bbox tidak hijau saat lewat garis
"""

import cv2
import os

def analyze_line_position(video_path, frame_sample=100):
    """Analyze frame properties dan line position"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video tidak ditemukan: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka video: {video_path}")
        return
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("📹 Video Analysis")
    print("="*60)
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.1f}s")
    print()
    
    # Current config
    LINE_POSITION = 300
    print("📍 Current Configuration")
    print("="*60)
    print(f"LINE_POSITION: {LINE_POSITION}px")
    print(f"Position %: {(LINE_POSITION/height)*100:.1f}% dari top")
    print(f"Offset: 40px")
    print(f"MIN_FRAMES_TO_COUNT: 5 frames (~{5/fps:.2f}s)")
    print()
    
    # Recommendations
    print("💡 Recommendations")
    print("="*60)
    if LINE_POSITION > height:
        print(f"⚠️  LINE_POSITION ({LINE_POSITION}) > video height ({height})")
        print(f"   Adjust to: {int(height * 0.6)} (60% dari atas)")
    elif LINE_POSITION < 100:
        print(f"⚠️  LINE_POSITION ({LINE_POSITION}) terlalu atas")
        print(f"   Adjust to: {int(height * 0.5)} (50% dari atas)")
    else:
        print(f"✅ LINE_POSITION ({LINE_POSITION}) OK untuk video {height}px")
    
    print()
    print("🔍 Sample Frame Analysis")
    print("="*60)
    
    # Jump to sample frames
    frame_indices = [100, total_frames//4, total_frames//2, (total_frames*3)//4]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames-1))
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Count horizontal lines (vehicles)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check area around line
        line_area_top = max(0, LINE_POSITION - 50)
        line_area_bottom = min(height, LINE_POSITION + 50)
        
        line_region = gray[line_area_top:line_area_bottom, :]
        
        # Detect edges
        edges = cv2.Canny(line_region, 50, 150)
        
        print(f"Frame {idx}:")
        print(f"  Line area ({line_area_top}-{line_area_bottom}): {np.count_nonzero(edges)} edge pixels")
    
    cap.release()
    
    print()
    print("📊 Configuration Suggestions")
    print("="*60)
    
    if height <= 360:
        print("Small video (≤360p):")
        line = int(height * 0.5)
        print(f"  LINE_POSITION = {line}")
        print(f"  MIN_FRAMES_TO_COUNT = 3")
        print(f"  MAX_DISTANCE = 40")
    elif height <= 480:
        print("Medium video (≤480p):")
        line = int(height * 0.6)
        print(f"  LINE_POSITION = {line}")
        print(f"  MIN_FRAMES_TO_COUNT = 5")
        print(f"  MAX_DISTANCE = 60")
    else:
        print("Large video (>480p):")
        line = int(height * 0.65)
        print(f"  LINE_POSITION = {line}")
        print(f"  MIN_FRAMES_TO_COUNT = 8")
        print(f"  MAX_DISTANCE = 80")
    
    print()

if __name__ == "__main__":
    import numpy as np
    
    print("🎬 YOLO Vehicle Counting - Diagnostic Tool")
    print("="*60)
    print()
    
    video_path = r"D:\backup\pktj\backend\yolo\input_videos\008.mp4"
    
    analyze_line_position(video_path)
    
    print("\n✨ Diagnostic Complete!")
    print("="*60)
    print("\nNotes:")
    print("- If box tidak hijau saat lewat garis, check LINE_POSITION")
    print("- If vehicles skipped, lower MIN_FRAMES_TO_COUNT")
    print("- If double counting, raise MIN_FRAMES_TO_COUNT")
    print("- Run again setelah adjust config")
