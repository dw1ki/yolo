#!/usr/bin/env python3
"""
YOLO Model Inference - Video 008 Trained Model
Menggunakan model yang dilatih dari dataset video 008
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Load model terlatih
MODEL_PATH = "models/best_video_008.pt"
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

def detect_in_video(video_path, output_path="output_inference.mp4", confidence=0.5):
    """
    Detect objects dalam video menggunakan model terlatih
    
    Args:
        video_path: Path ke video input
        output_path: Path untuk video output
        confidence: Confidence threshold (0-1)
    """
    print(f"\n🎬 Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka video: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    detections_total = 0
    
    print(f"   Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=confidence, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Count detections
        if results[0].boxes is not None:
            detections_total += len(results[0].boxes)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    # Release
    cap.release()
    out.release()
    
    print(f"\n✅ Inference selesai!")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total detections: {detections_total}")
    print(f"   Output video: {output_path}")

def detect_in_image(image_path, output_path=None, confidence=0.5):
    """
    Detect objects dalam gambar
    
    Args:
        image_path: Path ke gambar
        output_path: Path untuk output gambar (opsional)
        confidence: Confidence threshold
    """
    print(f"\n🖼️  Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        return
    
    # Run inference
    results = model(image_path, conf=confidence, verbose=False)
    
    # Plot results
    annotated_image = results[0].plot()
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Output saved to: {output_path}")
    
    # Print detections
    if results[0].boxes is not None:
        print(f"✅ Detections: {len(results[0].boxes)}")
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls)
            confidence_score = float(box.conf)
            class_name = model.names[class_id]
            print(f"   {i+1}. {class_name} (confidence: {confidence_score:.2f})")
    else:
        print("❌ No detections found")
    
    return results[0]

def batch_detect_images(image_dir, output_dir=None, confidence=0.5):
    """
    Detect objects dalam semua gambar di folder
    
    Args:
        image_dir: Path ke folder gambar
        output_dir: Path untuk output folder
        confidence: Confidence threshold
    """
    print(f"\n📁 Batch processing images from: {image_dir}")
    
    if not os.path.isdir(image_dir):
        print(f"❌ Directory tidak ditemukan: {image_dir}")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = list(Path(image_dir).glob("*.jpg")) + \
                  list(Path(image_dir).glob("*.jpeg")) + \
                  list(Path(image_dir).glob("*.png"))
    
    print(f"   Found {len(image_files)} images")
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n   [{idx}/{len(image_files)}] {image_path.name}")
        
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"detected_{image_path.name}")
        
        detect_in_image(str(image_path), output_path, confidence)

# Class mapping
CLASSES = {
    0: "Car",
    1: "Bus",
    2: "Truck"
}

def print_model_info():
    """Print model information"""
    print("\n" + "="*50)
    print("📋 Model Information")
    print("="*50)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Size: 49.61 MB")
    print(f"Training Date: March 23, 2026")
    print(f"Training Data: Video 008 (641 frames)")
    print(f"Training Results:")
    print(f"  - mAP50: 0.498 (49.8%)")
    print(f"  - mAP50-95: 0.302 (30.2%)")
    print(f"  - Car mAP50: 0.981 ⭐")
    print(f"  - Inference speed: 300.7ms per image")
    print(f"\nClasses ({len(model.names)} total):")
    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Print model info
    print_model_info()
    
    # Example usage (uncomment to use):
    
    # 1. Detect dalam video
    # detect_in_video(
    #     video_path="input_videos/008.mp4",
    #     output_path="output_videos/008_detected.mp4",
    #     confidence=0.5
    # )
    
    # 2. Detect dalam gambar
    # detect_in_image(
    #     image_path="path/to/image.jpg",
    #     output_path="output_detected.jpg",
    #     confidence=0.5
    # )
    
    # 3. Batch detect gambar
    # batch_detect_images(
    #     image_dir="input_images/",
    #     output_dir="output_images/",
    #     confidence=0.5
    # )
    
    print("💡 Gunakan fungsi di atas untuk melakukan inference!")
    print("   - detect_in_video() untuk video")
    print("   - detect_in_image() untuk gambar")
    print("   - batch_detect_images() untuk batch gambar")
