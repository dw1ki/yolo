# 📊 YOLO MODEL EVALUATION GUIDE - STEP BY STEP

**Panduan Lengkap Menguji Akurasi Model YOLOv8 Deteksi Kendaraan**

---

## TABLE OF CONTENTS

1. [Quick Start (5 menit)](#quick-start)
2. [Detailed Evaluation (30 menit)](#detailed-evaluation)
3. [Understanding Metrics](#understanding-metrics)
4. [Interpreting Results](#interpreting-results)
5. [Performance Optimization](#performance-optimization)

---

## QUICK START ⚡

### Option 1: Gunakan YOLO Built-in Validation (Recommended)

```bash
# Jika punya validation dataset yang terstruktur
cd backend/yolo

python -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or runs/detect/train/weights/best.pt

results = model.val(
    data='data.yaml',
    imgsz=1280,
    batch=4
)

print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
"
```

**Output:**
```
val: Scanning /path/to/val/images... 300 images, 0 backgrounds, 0 corrupt
        Class     Images     Targets           P           R      mAP50   mAP50-95
          all        300        1500       0.856       0.892       0.875       0.654
        mobil        300        600       0.876       0.912       0.895       0.673
          bus        300        400       0.834       0.872       0.854       0.632
         truk        300        500       0.847       0.882       0.866       0.647
```

### Option 2: Test pada Single Image

```bash
python -c "
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

# Predict
results = model.predict(source='test_image.jpg', conf=0.5)

# Display
result = results[0]
im_array = result.plot()
Image.fromarray(im_array[..., ::-1]).save('result.jpg')

# Print detections
for box in result.boxes:
    print(f'{model.names[int(box.cls)]}: {float(box.conf):.2f}')
"
```

---

## DETAILED EVALUATION 📊

### Step 1: Persiapkan Dataset Test

```
data/
├── images/
│   ├── train/    (70% - untuk training)
│   └── val/      (30% - untuk validation/testing)
└── labels/
    ├── train/    (YOLO format: .txt files)
    └── val/
```

**Format file label (`.txt`):**
```
<class_id> <x_center> <y_center> <width> <height>
0 0.456 0.632 0.234 0.567
1 0.234 0.789 0.123 0.456
```

### Step 2: Run Comprehensive Evaluation

```bash
cd backend/yolo

python evaluate_model.py
```

**Script ini akan:**
- ✅ Run YOLO built-in validation
- ✅ Calculate precision, recall, mAP50, mAP50-95
- ✅ Generate confusion matrix
- ✅ Analyze per-class performance
- ✅ Test pada video (vehicle counting)
- ✅ Generate visualization plots

**Output:**
```
============================================================
🧪 PHASE 1: YOLO BUILT-IN VALIDATION
============================================================
Validating...
Computing metrics...

📊 OVERALL METRICS
--------------------------------------------------
Precision (P):     0.8456
Recall (R):        0.8924
mAP50:             0.8654
mAP50-95:          0.5432

📊 PER-CLASS METRICS
--------------------------------------------------

Mobil:
  TP: 450, FP: 32, FN: 28
  Precision: 0.8821
  Recall:    0.9412
  F1-Score:  0.9109

Bus:
  TP: 120, FP: 18, FN: 15
  Precision: 0.8696
  Recall:    0.8889
  F1-Score:  0.8791

Truk:
  TP: 180, FP: 22, FN: 20
  Precision: 0.8910
  Recall:    0.9000
  F1-Score:  0.8955

✅ EVALUATION COMPLETE!
```

### Step 3: Video Analysis dengan Vehicle Counting

```python
from evaluate_model import YOLOEvaluator

evaluator = YOLOEvaluator("yolov8n.pt", "data.yaml")

# Analyze video dan count vehicles
stats = evaluator.evaluate_video(
    video_path="sample_video.mp4",
    output_video="output_with_detection.mp4"  # Optional
)

print(f"Total vehicles detected: {stats['vehicles_detected']}")
print(f"By class: {stats['by_class']}")
```

---

## UNDERSTANDING METRICS 🎯

### 1. PRECISION (P) - Accuracy of Positive Predictions

```
Formula: TP / (TP + FP)

Penjelasan:
- Dari 100 kendaraan yang MODEL DETEKSI
- Berapa banyak yang BENAR?

Example:
  Jika model deteksi 100 mobil dan 80 benar
  Precision = 80/100 = 0.80 (80%)

Target: > 0.75 (75%)
Trade-off: High precision = fewer false positives, lebih ke-underdetect
```

### 2. RECALL (R) - Coverage of Actual Objects

```
Formula: TP / (TP + FN)

Penjelasan:
- Dari SEMUA kendaraan yang ada di video
- Berapa % yang MODEL TERDETEKSI?

Example:
  Jika ada 100 kendaraan dan model deteksi 85
  Recall = 85/100 = 0.85 (85%)

Target: > 0.80 (80%)
Trade-off: High recall = fewer miss, tapi lebih false positive
```

### 3. mAP50 - Average Precision at IoU 0.5

```
Formula: Average precision across confidence thresholds
         at IoU threshold 0.5

Penjelasan:
- IoU (Intersection over Union) = overlap antara predicted 
  dan ground truth bounding box
- 0.5 artinya minimal 50% overlap dianggap "correct"

Contoh IoU:
  IoU = 0.7 (70% overlap) → CORRECT ✓
  IoU = 0.3 (30% overlap) → WRONG ✗

Target: > 0.80 (80%)
Gunakan untuk: Overall model evaluation
```

### 4. mAP50-95 - Strict Average Precision

```
Formula: Average mAP pada IoU 0.5, 0.55, 0.60, ..., 0.95

Penjelasan:
- Lebih strict dari mAP50
- Memerlukan bounding box lebih akurat

Target: > 0.55 (55%)
Gunakan untuk: Production quality check (bounding box presisi)
```

### 5. F1-Score - Balance antara Precision & Recall

```
Formula: 2 * (P * R) / (P + R)

Penjelasan:
- Score 0-1 yang balance precision dan recall
- F1 = 1.0 adalah perfect (precision=1.0, recall=1.0)
- Digunakan untuk: Overall model performance

Target: > 0.78 (78%)

Example:
  P=0.8, R=0.8 → F1 = 2*(0.8*0.8)/(0.8+0.8) = 0.80
  P=0.9, R=0.7 → F1 = 2*(0.9*0.7)/(0.9+0.7) = 0.787
```

### Comparison Matrix

| Metric | Formula | Target | Use Case |
|--------|---------|--------|----------|
| **Precision** | TP/(TP+FP) | >0.75 | Minimize false alarms |
| **Recall** | TP/(TP+FN) | >0.80 | Minimize missed detections |
| **F1-Score** | 2(PR)/(P+R) | >0.78 | Balanced evaluation |
| **mAP50** | Area under PR curve @ IoU0.5 | >0.80 | Overall performance |
| **mAP50-95** | Average mAP @ IoU 0.5-0.95 | >0.55 | Strict evaluation |

---

## INTERPRETING RESULTS 📈

### Skenario 1: "Semua Metrics Bagus"

```
Precision: 0.87 ✅
Recall: 0.89 ✅
mAP50: 0.88 ✅
mAP50-95: 0.65 ✅
```

**Interpretasi:** Model siap production! ✨

### Skenario 2: "High Precision, Low Recall"

```
Precision: 0.92 ✅ (deteksi sedikit tapi akurat)
Recall: 0.65 ❌ (melewatkan banyak kendaraan)
```

**Problem:** Model terlalu konservatif, missed detections

**Solusi:**
```python
# Turun confidence threshold
results = model.predict(source='video.mp4', conf=0.3)  # dari 0.5 → 0.3

# Atau retrain dengan lebih banyak negative samples
# Atau tambah augmentation
```

### Skenario 3: "Low Precision, High Recall"

```
Precision: 0.65 ❌ (banyak false positives)
Recall: 0.92 ✅ (deteksi semua, termasuk false)
```

**Problem:** Model terlalu agresif, false alarms

**Solusi:**
```python
# Naik confidence threshold
results = model.predict(source='video.mp4', conf=0.7)  # dari 0.5 → 0.7

# Atau retrain dengan better quality labels
# Atau gunakan NMS dengan IoU lebih tinggi
```

### Skenario 4: "Low Performance Semua"

```
Precision: 0.52 ❌
Recall: 0.58 ❌
mAP50: 0.48 ❌
```

**Possible Causes:**
1. Dataset terlalu kecil (< 300 images per class)
2. Data distribution mismatch (training vs real)
3. Poor label quality (annotation errors)
4. Hyperparameter tidak cocok
5. Model architecture terlalu kecil

**Solusi:**
```python
# Add more diverse data
# Increase augmentation
# We use yolov8n (nano model)
# Fine-tune training parameters
```

---

## PERFORMANCE OPTIMIZATION 🚀

### 1. Confidence Threshold Tuning

```python
# Analyze optimal threshold
from evaluate_model import YOLOEvaluator

evaluator = YOLOEvaluator("best.pt")
results = evaluator.analyze_confidence_threshold("data/images/val")

# results[0.5] = {'precision': 0.85, 'recall': 0.87, 'f1': 0.86}
# results[0.6] = {'precision': 0.89, 'recall': 0.82, 'f1': 0.85}
# results[0.4] = {'precision': 0.81, 'recall': 0.91, 'f1': 0.86}

# Choose threshold dengan highest F1-score
best_threshold = max(results.items(), key=lambda x: x[1]['f1_score'])
print(f"Optimal threshold: {best_threshold[0]}")
```

### 2. Improve Per-Class Performance

```python
# If one class (e.g., "bus") punya low recall:

# Option 1: Collect more bus data
# Option 2: Increase augmentation untuk bus
# Option 3: Assign higher loss weight untuk bus

model.train(
    data='data.yaml',
    epochs=60,
    cls_weight=[1.0, 2.0, 1.0],  # 2x weight untuk bus (class 1)
)
```

### 3. Model Architecture Selection

```python
# Jika memory/speed adalah constraint:

from ultralytics import YOLO

# Nano - fastest, smallest (~6MB)
model = YOLO('yolov8n.pt')

# Small - balanced (~13MB)
model = YOLO('yolov8s.pt')

# Medium - better accuracy (~25MB)
model = YOLO('yolov8n.pt')

# Large - best accuracy (~52MB)
model = YOLO('yolov8l.pt')

# Trade-off: Accuracy vs Speed vs Memory
# Nano: 90 FPS, 80% accuracy
# Small: 70 FPS, 82% accuracy
# Medium: 40 FPS, 85% accuracy
# Large: 20 FPS, 87% accuracy
```

### 4. Inference Optimization

```python
# Export model untuk production (faster)

model = YOLO('best.pt')

# Export to different formats
model.export(format='onnx')   # CPU-friendly
model.export(format='tflite')  # Mobile
model.export(format='torchscript')  # PyTorch

# Use exported model
onnx_model = YOLO('best.onnx')
results = onnx_model.predict(source='video.mp4')
```

---

## REAL-TIME MONITORING 📊

```python
# Setup WandB untuk tracking (optional)

import wandb
from ultralytics import YOLO

wandb.login()  # Sign up di wandb.ai

model = YOLO('best.pt')

# Inference dengan logging
results = model.predict(
    source='video.mp4',
    conf=0.5,
    save_txt=True,  # Save predictions
    save_conf=True
)

# Log ke WandB
wandb.log({'detections': len(results)})
```

---

## PRODUCTION CHECKLIST ✅

Sebelum deploy ke production:

```
✅ Metric Checklist:
  - [ ] Precision > 0.75
  - [ ] Recall > 0.80
  - [ ] mAP50 > 0.80
  - [ ] F1-Score > 0.78
  - [ ] mAP50-95 > 0.55 (jika perlu strict)

✅ Performance Checklist:
  - [ ] Inference time < 100ms (30 FPS)
  - [ ] Memory usage < 2GB
  - [ ] GPU utilization < 80%
  - [ ] Tested on diverse videos (day, night, rain, etc)

✅ Data Checklist:
  - [ ] Test set > 300 images
  - [ ] All classes represented
  - [ ] Confidence threshold optimized
  - [ ] Per-class metrics reviewed

✅ Output Checklist:
  - [ ] evaluation_report.md generated
  - [ ] confusion_matrix_results.png reviewed
  - [ ] output_annotated.mp4 quality checked
  - [ ] benchmark.txt with timing info
```

---

## TROUBLESHOOTING ❓

### Q1: "Error: No such file or directory: data.yaml"
```
A: Pastikan path relatif ke backend/yolo directory
   cd backend/yolo
   python evaluate_model.py
```

### Q2: "CUDA out of memory"
```
A: Turun batch size atau image size
   model.val(batch=2, imgsz=640)
```

### Q3: "Metrics tidak improve setelah training"
```
A: Possible causes:
   - Dataset terlalu kecil (need > 500 per class)
   - Data quality issue (bad annotations)
   - Learning rate terlalu tinggi/rendah
   - Model underfitting → use larger model
```

### Q4: "Video analysis terlalu lambat"
```
A: Options:
   - Gunakan smaller model: yolov8n.pt
   - Turun imgsz: imgsz=640
   - Skip frames: process every Nth frame
   - Disable writing output video
```

---

## NEXT STEPS 🚀

1. ✅ **Run evaluation_model.py** - Get baseline metrics
2. ✅ **Review per-class performance** - Identify weak classes
3. ✅ **Optimize confidence threshold** - Tune precision/recall tradeoff
4. ✅ **Test on production video** - Real-world scenarios
5. ✅ **Deploy best model** - Update API dengan best.pt
6. ✅ **Monitor in production** - Track metrics over time

---

## REFERENCES

- YOLOv8 Docs: https://docs.ultralytics.com/
- mAP Explanation: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
- Object Detection Metrics: https://github.com/rafaelpadilla/Object-Detection-Metrics

---

**Last Updated**: February 13, 2026  
**Status**: ✅ Complete & Ready to Use  
**Author**: System Evaluation Team
