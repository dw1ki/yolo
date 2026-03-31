"""
=============================================================================
COMPREHENSIVE YOLO ACCURACY EVALUATION SYSTEM
=============================================================================
Sistem evaluasi lengkap untuk menguji akurasi model YOLOv8 deteksi kendaraan.

Features:
- Metrics calculation: Precision, Recall, mAP50, mAP50-95, F1-Score
- Per-class evaluation: Mobil, Bus, Truk
- Confusion matrix generation
- Video analysis dengan vehicle counting
- Lane classification accuracy (if available)
- Comprehensive reporting & visualization

Author: System Evaluator
Date: 2026-02-13
=============================================================================
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


class YOLOEvaluator:
    """Comprehensive YOLO Model Evaluator"""
    
    def __init__(self, model_path: str, data_yaml: str = "data.yaml"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model (best.pt)
            data_yaml: Path to data.yaml configuration
        """
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.classes = self._load_classes()
        self.results = {
            'metrics': {},
            'per_class': {},
            'confusion_matrix': None,
            'detections': [],
            'timing': {}
        }
        
    def _load_classes(self) -> Dict[int, str]:
        """Load class names from data.yaml"""
        import yaml
        with open(self.data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']
    
    # ============================================================
    # 1. INTEGRATED VALIDATION (Using YOLO built-in)
    # ============================================================
    
    def validate_with_yolo(self) -> Dict:
        """
        Use YOLOv8 built-in validation
        Best for: Dataset dengan proper train/val split
        """
        print("\n" + "="*60)
        print("🧪 PHASE 1: YOLO BUILT-IN VALIDATION")
        print("="*60)
        
        results = self.model.val(
            data=self.data_yaml,
            imgsz=1280,
            batch=4,
            device=0,
            # conf=0.25,  # Default YOLO confidence
            # iou=0.6,    # NMS IoU threshold
            plots=True  # Generate validation plots
        )
        
        # Extract metrics
        metrics = {
            'precision': float(results.box.mp),  # mAP50
            'recall': float(results.box.mr),
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'fitness': float(results.box.fitness())
        }
        
        self.results['metrics'] = metrics
        self._print_metrics(metrics)
        
        return metrics
    
    # ============================================================
    # 2. TEST SET EVALUATION (For custom test images/videos)
    # ============================================================
    
    def evaluate_test_dataset(self, test_dir: str) -> Dict:
        """
        Evaluate pada test dataset (images with ground truth annotations)
        
        Args:
            test_dir: Directory dengan struktur:
                      test_dir/images/*.jpg
                      test_dir/labels/*.txt (YOLO format)
        """
        print("\n" + "="*60)
        print("🧪 PHASE 2: CUSTOM TEST DATASET EVALUATION")
        print("="*60)
        
        test_path = Path(test_dir)
        images = list(test_path.glob("images/*.jpg"))
        
        if not images:
            print("❌ No images found in test_dir/images/")
            return {}
        
        print(f"📊 Found {len(images)} test images")
        
        # Run inference
        predictions = self.model.predict(
            source=str(test_path / "images"),
            conf=0.5,
            iou=0.45,
            imgsz=1280,
            device=0,
            verbose=False
        )
        
        # Calculate metrics
        tp, fp, fn = 0, 0, 0
        per_class = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for pred, img_file in zip(predictions, images):
            # Load ground truth
            label_file = test_path / "labels" / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                continue
            
            gt_boxes = self._load_yolo_annotations(label_file)
            pred_boxes = self._extract_predictions(pred)
            
            # Match predictions to ground truth
            matches = self._match_boxes(pred_boxes, gt_boxes)
            
            # Update metrics
            for match in matches:
                if match['matched']:
                    tp += 1
                    per_class[self.classes[match['class']]]['tp'] += 1
                else:
                    fp += 1
                    per_class[self.classes[match['class']]]['fp'] += 1
            
            for gt in gt_boxes:
                if not any(m['matched'] and m['gt_id'] == id(gt) for m in matches):
                    fn += 1
                    per_class[self.classes[gt['class']]]['fn'] += 1
        
        # Calculate metrics
        metrics = self._compute_metrics(tp, fp, fn)
        self.results['per_class'] = per_class
        self.results['metrics'] = metrics
        
        self._print_metrics(metrics)
        self._print_per_class_metrics(per_class)
        
        return metrics
    
    # ============================================================
    # 3. VIDEO EVALUATION WITH VEHICLE COUNTING
    # ============================================================
    
    def evaluate_video(self, video_path: str, output_video: Optional[str] = None) -> Dict:
        """
        Evaluate akurasi pada video dengan real-time vehicle counting
        
        Args:
            video_path: Path ke video file
            output_video: Optional output video dengan annotasi
        """
        print("\n" + "="*60)
        print("🎬 PHASE 3: VIDEO EVALUATION WITH TRACK & COUNT")
        print("="*60)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📽️  Video: {Path(video_path).name}")
        print(f"   Duration: {frame_count/fps:.1f}s ({frame_count} frames @ {fps:.1f} fps)")
        
        # Setup output video if needed
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, 
                                 (int(cap.get(3)), int(cap.get(4))))
        
        # Vehicle tracking
        vehicles = defaultdict(lambda: {
            'count': 0, 'classes': defaultdict(int), 'frames_seen': [],
            'confidences': []
        })
        frame_vehicles = defaultdict(list)
        
        # Per-class count
        total_by_class = defaultdict(int)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            results = self.model.predict(
                source=frame,
                conf=0.5,
                iou=0.45,
                imgsz=1280,
                device=0,
                verbose=False
            )
            
            # Extract detections
            detections = results[0].boxes.data.cpu().numpy()
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = self.classes[int(cls)]
                total_by_class[class_name] += 1
                
                frame_vehicles[frame_idx].append({
                    'class': class_name,
                    'conf': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
                
                # Draw on frame if output video
                if output_video:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame
            if output_video:
                out.write(frame)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"  🔄 Processed {frame_idx}/{frame_count} frames...")
        
        cap.release()
        if output_video:
            out.release()
            print(f"✅ Output video saved: {output_video}")
        
        # Summary statistics
        video_stats = {
            'total_frames': frame_idx,
            'vehicles_detected': sum(len(v) for v in frame_vehicles.values()),
            'by_class': dict(total_by_class),
            'avg_per_frame': sum(len(v) for v in frame_vehicles.values()) / frame_idx,
            'avg_confidence': np.mean([
                det['conf'] for dets in frame_vehicles.values() for det in dets
            ])
        }
        
        self.results['video_stats'] = video_stats
        self._print_video_stats(video_stats)
        
        return video_stats
    
    # ============================================================
    # 4. CONFUSION MATRIX GENERATION
    # ============================================================
    
    def generate_confusion_matrix(self, test_dir: str) -> np.ndarray:
        """Generate confusion matrix for classifications"""
        print("\n" + "="*60)
        print("🔲 PHASE 4: CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        n_classes = len(self.classes)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        test_path = Path(test_dir)
        images = list(test_path.glob("images/*.jpg"))
        
        for img_file in images:
            # Ground truth
            label_file = test_path / "labels" / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            gt_boxes = self._load_yolo_annotations(label_file)
            
            # Predictions
            frame = cv2.imread(str(img_file))
            results = self.model.predict(frame, conf=0.5, verbose=False)
            pred_boxes = self._extract_predictions(results[0])
            
            # Match and update matrix
            matches = self._match_boxes(pred_boxes, gt_boxes)
            
            for match in matches:
                if match['matched']:
                    pred_class = match['class']
                    gt_class = match['gt_class']
                    conf_matrix[gt_class][pred_class] += 1
        
        self.results['confusion_matrix'] = conf_matrix
        self._plot_confusion_matrix(conf_matrix)
        
        return conf_matrix
    
    # ============================================================
    # 5. CONFIDENCE THRESHOLD ANALYSIS
    # ============================================================
    
    def analyze_confidence_threshold(self, test_dir: str) -> Dict:
        """
        Analyze effect of different confidence thresholds on precision/recall
        """
        print("\n" + "="*60)
        print("📈 PHASE 5: CONFIDENCE THRESHOLD ANALYSIS")
        print("="*60)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {t: {'precision': 0, 'recall': 0, 'f1': 0} for t in thresholds}
        
        for conf_thresh in thresholds:
            tp, fp, fn = 0, 0, 0
            
            test_path = Path(test_dir)
            images = list(test_path.glob("images/*.jpg"))
            
            for img_file in images:
                label_file = test_path / "labels" / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue
                
                gt_boxes = self._load_yolo_annotations(label_file)
                
                frame = cv2.imread(str(img_file))
                predictions = self.model.predict(
                    frame, conf=conf_thresh, verbose=False
                )
                pred_boxes = self._extract_predictions(predictions[0])
                
                matches = self._match_boxes(pred_boxes, gt_boxes)
                
                for match in matches:
                    if match['matched']:
                        tp += 1
                    else:
                        fp += 1
                
                for _ in gt_boxes:
                    if not any(m['matched'] and m['gt_id'] == _ for m in matches):
                        fn += 1
            
            metrics = self._compute_metrics(tp, fp, fn)
            results[conf_thresh] = metrics
        
        self._plot_confidence_analysis(results)
        return results
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    def _load_yolo_annotations(self, label_file: Path) -> List[Dict]:
        """Load YOLO format annotations"""
        boxes = []
        with open(label_file) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = parts[1:5]
                    boxes.append({
                        'class': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        return boxes
    
    def _extract_predictions(self, result) -> List[Dict]:
        """Extract detections from YOLO result"""
        boxes = []
        if result.boxes is None:
            return boxes
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            boxes.append({
                'class': cls,
                'conf': conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'x_center': (x1 + x2) / 2,
                'y_center': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1
            })
        
        return boxes
    
    def _match_boxes(self, pred_boxes: List, gt_boxes: List, iou_thresh: float = 0.5) -> List:
        """Match predicted boxes to ground truth using IoU"""
        matches = []
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                iou = self._calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            matched = best_iou >= iou_thresh
            matches.append({
                'matched': matched,
                'iou': best_iou,
                'class': pred['class'],
                'gt_class': gt_boxes[best_gt_idx]['class'] if best_gt_idx >= 0 else -1,
                'conf': pred['conf'],
                'gt_id': best_gt_idx
            })
        
        return matches
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate IoU between two boxes"""
        # Convert to pixel coordinates if needed
        if 'x1' in box1:
            x1_min, y1_min, x1_max, y1_max = box1['x1'], box1['y1'], box1['x2'], box1['y2']
        else:
            # YOLO normalized format - would need image size to convert
            return 0
        
        x2_min = box2['x_center'] - box2['width'] / 2
        y2_min = box2['y_center'] - box2['height'] / 2
        x2_max = box2['x_center'] + box2['width'] / 2
        y2_max = box2['y_center'] + box2['height'] / 2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        intersection = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _compute_metrics(self, tp: int, fp: int, fn: int) -> Dict:
        """Compute precision, recall, F1"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_detections': tp + fp,
            'total_ground_truth': tp + fn
        }
    
    # ============================================================
    # REPORTING & VISUALIZATION
    # ============================================================
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics summary"""
        print("\n📊 OVERALL METRICS")
        print("-" * 50)
        
        if 'mAP50' in metrics:
            print(f"Precision (P):     {metrics.get('precision', 0):.4f}")
            print(f"Recall (R):        {metrics.get('recall', 0):.4f}")
            print(f"mAP50:             {metrics['mAP50']:.4f}")
            print(f"mAP50-95:          {metrics['mAP50_95']:.4f}")
        else:
            print(f"Precision (P):     {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
            print(f"Recall (R):        {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
            print(f"F1-Score:          {metrics.get('f1_score', 0):.4f}")
    
    def _print_per_class_metrics(self, per_class: Dict):
        """Print per-class metrics"""
        print("\n📊 PER-CLASS METRICS")
        print("-" * 50)
        
        for class_name, counts in per_class.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            metrics = self._compute_metrics(tp, fp, fn)
            print(f"\n{class_name}:")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    def _print_video_stats(self, stats: Dict):
        """Print video analysis statistics"""
        print("\n📊 VIDEO ANALYSIS RESULTS")
        print("-" * 50)
        print(f"Total Frames:          {stats['total_frames']}")
        print(f"Vehicles Detected:     {stats['vehicles_detected']}")
        print(f"Avg per Frame:         {stats['avg_per_frame']:.2f}")
        print(f"Avg Confidence:        {stats['avg_confidence']:.4f}")
        print(f"\nDetails by Class:")
        for class_name, count in stats['by_class'].items():
            print(f"  {class_name:15} {count:5} vehicles")
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=self.classes.values(),
                   yticklabels=self.classes.values())
        plt.title('Confusion Matrix - Vehicle Detection')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_results.png', dpi=150)
        print("\n✅ Confusion matrix saved: confusion_matrix_results.png")
        plt.close()
    
    def _plot_confidence_analysis(self, results: Dict):
        """Plot confidence threshold analysis"""
        thresholds = list(results.keys())
        precisions = [results[t]['precision'] for t in thresholds]
        recalls = [results[t]['recall'] for t in thresholds]
        f1_scores = [results[t]['f1_score'] for t in thresholds]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 's-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, '^-', label='F1-Score', linewidth=2)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Confidence Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(recalls, precisions, 'o-', linewidth=2)
        for i, t in enumerate(thresholds):
            plt.annotate(f'{t:.1f}', (recalls[i], precisions[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis_results.png', dpi=150)
        print("✅ Confidence analysis saved: confidence_analysis_results.png")
        plt.close()
    
    def generate_report(self, output_file: str = "evaluation_report.md"):
        """Generate comprehensive evaluation report"""
        with open(output_file, 'w') as f:
            f.write("# 📋 YOLO MODEL EVALUATION REPORT\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            if self.results['metrics']:
                f.write("## Overall Metrics\n")
                for key, value in self.results['metrics'].items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.4f}\n")
            
            # Per-class metrics
            if self.results['per_class']:
                f.write("\n## Per-Class Performance\n")
                for class_name, counts in self.results['per_class'].items():
                    f.write(f"\n### {class_name}\n")
                    for key, value in counts.items():
                        f.write(f"- {key}: {value}\n")
        
        print(f"\n✅ Report saved: {output_file}")


# =============================================================================
# MAIN EVALUATION WORKFLOW
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   YOLO VEHICLE DETECTION - COMPREHENSIVE EVALUATION       ║
    ║   Testing Akurasi: Mobil, Bus, Truk                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Setup paths
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    DATA_YAML = "data.yaml"
    TEST_VIDEO = "test.mp4"  # Replace with your video
    TEST_DIR = "data/images/val"  # Validation dataset
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(MODEL_PATH, DATA_YAML)
    
    # Run evaluation phases
    print("\n🚀 STARTING COMPREHENSIVE EVALUATION...\n")
    
    # Phase 1: Built-in YOLO validation
    print("Step 1/4: YOLO Built-in Validation")
    try:
        evaluator.validate_with_yolo()
    except Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # Phase 2: Custom test dataset
    print("\nStep 2/4: Custom Test Dataset")
    try:
        evaluator.evaluate_test_dataset(TEST_DIR)
    except Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # Phase 3: Video evaluation
    print("\nStep 3/4: Video Evaluation")
    try:
        if os.path.exists(TEST_VIDEO):
            evaluator.evaluate_video(TEST_VIDEO, output_video="output_annotated.mp4")
        else:
            print(f"⚠️  Video not found: {TEST_VIDEO}")
    except Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # Phase 4: Confusion matrix
    print("\nStep 4/4: Confusion Matrix Analysis")
    try:
        evaluator.generate_confusion_matrix(TEST_DIR)
    except Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # Generate report
    evaluator.generate_report()
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\nOutputs generated:")
    print("- confusion_matrix_results.png")
    print("- confidence_analysis_results.png")
    print("- evaluation_report.md")
    print("- output_annotated.mp4 (if video provided)")
