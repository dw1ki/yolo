#!/usr/bin/env python3
"""Test video processing with MJPG codec"""
import cv2, numpy as np, sys, os

VIDEO_PATH = "/mnt/data2/pktj/backend/yolo/video/siang-singkat.mp4"
OUTPUT_PATH = "/tmp/test_output.avi"

print("=" * 60)
print("🧪 Testing Video Processing Fixes with MJPG Codec")
print("=" * 60)

# Clean previous output
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)
    print(f"✅ Removed old output file")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

fps_orig = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

in_duration = total_frames / int(fps_orig)
print(f"\n📹 INPUT VIDEO:")
print(f"   File: {os.path.basename(VIDEO_PATH)}")
print(f"   Resolution: {w}x{h}")
print(f"   Frames: {total_frames}")
print(f"   FPS: {fps_orig} (original) → {int(fps_orig)} (int)")
print(f"   Duration: {in_duration:.2f}s")

fps = int(fps_orig)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

if not out_writer.isOpened():
    print(f"\n❌ VideoWriter failed to open!")
    cap.release()
    sys.exit(1)

print(f"\n🎬 PROCESSING:")
print(f"   Output: {OUTPUT_PATH}")
print(f"   Codec: MJPG")
print(f"   FPS: {fps}")

frame_count = 0
written_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Add simple annotation
    cv2.line(frame, (0, 300), (w, 300), (0, 0, 255), 3)
    
    # Prepare frame for writing
    frame_write = np.ascontiguousarray(frame.astype(np.uint8))
    
    # Write frame (MJPG doesn't return bool, so we just write)
    out_writer.write(frame_write)
    written_count += 1
    
    if frame_count % 100 == 0:
        print(f"   Processed: {frame_count}/{total_frames} frames...")

cap.release()
out_writer.release()

print(f"\n✅ WRITTEN: {written_count} frames")

# Verify output
print(f"\n📊 OUTPUT VERIFICATION:")
out_cap = cv2.VideoCapture(OUTPUT_PATH)
out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_fps = int(out_cap.get(cv2.CAP_PROP_FPS))
out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_cap.release()

out_duration = out_frames / out_fps if out_fps > 0 else 0
diff = abs(in_duration - out_duration)

print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
print(f"   Resolution: {out_w}x{out_h}")
print(f"   Frames: {out_frames}")
print(f"   FPS: {out_fps}")
print(f"   Duration: {out_duration:.2f}s")

print(f"\n📈 COMPARISON:")
print(f"   Input duration:  {in_duration:.2f}s")
print(f"   Output duration: {out_duration:.2f}s")
print(f"   Difference: {diff:.2f}s")

if diff < 0.5:
    print(f"\n✅ SUCCESS! Duration preserved within 0.5s tolerance")
else:
    print(f"\n❌ FAILED! Duration changed by {diff:.2f}s")

print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)

print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)
