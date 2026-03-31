# 📊 YOLO ACCURACY TESTING - DATA FORMAT & IMPLEMENTASI

**Penjelasan lengkap: Apa data yang dihasilkan saat test akurasi YOLO?**

---

## 1. CONFUSION MATRIX - DATA FORMAT

Confusion Matrix menunjukkan: **Apa yang model prediksi vs Apa yang sebenarnya**

### Format Tabel:

```
CONFUSION MATRIX (Normalized):

                  Predicted
            Mobil    Bus     Truk    Total
          ┌────────┬────────┬────────┐
Actual:   │        │        │        │
  Mobil   │ 1890   │  60    │  50    │ = 2000 (Recall Mobil: 94.5%)
  Bus     │  95    │ 1700   │ 105    │ = 1900 (Recall Bus: 89.5%)
  Truk    │  70    │ 130    │ 1700   │ = 1900 (Recall Truk: 89.5%)
          └────────┴────────┴────────┘
            Precision untuk prediksi:
              90%      93%      94%
```

### Yang Artinya:

```
Diagonal (benar): 1890 + 1700 + 1700 = 5290 (True Positives)
Off-diagonal (salah): 60 + 50 + 95 + 105 + 70 + 130 = 510 (Salah klasifikasi)

Total deteksi: 5800
Akurasi: 5290 / 5800 = 91.2%
```

---

## 2. PER-CLASS METRICS - DATA FORMAT

Metrik untuk setiap jenis kendaraan.

### Format Data (JSON):

```json
{
  "metrics": {
    "mobil": {
      "precision": 0.8850,
      "recall": 0.9450,
      "mAP50": 0.8950,
      "mAP50-95": 0.7420,
      "f1_score": 0.9140,
      "true_positives": 1890,
      "false_positives": 230,
      "false_negatives": 110
    },
    "bus": {
      "precision": 0.9320,
      "recall": 0.8950,
      "mAP50": 0.8420,
      "mAP50-95": 0.6820,
      "f1_score": 0.8630,
      "true_positives": 1700,
      "false_positives": 125,
      "false_negatives": 200
    },
    "truk": {
      "precision": 0.9420,
      "recall": 0.8950,
      "mAP50": 0.8650,
      "mAP50-95": 0.7150,
      "f1_score": 0.8780,
      "true_positives": 1700,
      "false_positives": 105,
      "false_negatives": 200
    }
  },
  "overall": {
    "precision": 0.9200,
    "recall": 0.9150,
    "mAP50": 0.8673,
    "mAP50-95": 0.7130,
    "f1_score": 0.9175,
    "total_tp": 5290,
    "total_fp": 460,
    "total_fn": 510,
    "total_predictions": 5750
  }
}
```

### Penjelasan Per-Class:

| Metrik | Arti |
|--------|------|
| **Precision** | Dari semua yang diprediksi "mobil", berapa % benar? |
| **Recall** | Dari semua mobil asli, berapa % terdeteksi? |
| **mAP50** | Rata-rata precision pada IoU≥0.5 (standar) |
| **mAP50-95** | Rata-rata precision pada IoU 0.5-0.95 (lebih ketat) |
| **F1-score** | Balance antara Precision & Recall |

---

## 3. CONFIDENCE THRESHOLD ANALYSIS

Menunjukkan: **Bagaimana akurasi berubah jika confidence threshold berbeda?**

### Format Data (CSV):

```csv
confidence_threshold,precision,recall,f1_score,num_detections
0.30,0.82,0.96,0.88,5800
0.40,0.85,0.93,0.89,5600
0.50,0.89,0.88,0.88,5200
0.60,0.92,0.82,0.87,4800
0.70,0.94,0.75,0.83,4200
0.80,0.96,0.65,0.78,3000
0.90,0.98,0.45,0.61,1500
```

### Visualisasi (ASCII):

```
Precision vs Recall vs Confidence

Precision ↑
0.98  │     ●
0.96  │    ●
0.94  │   ●
0.92  │  ●
0.90  │ ●
0.88  │●
      └─────────────────→
      0.3  0.5  0.7  0.9  Confidence
      
Recall ↓ (berbanding terbalik dengan Precision)
```

### Interpretasi:

```
Confidence 0.5 = Balance Point (Rekomendasi)
├─ Precision: 0.89 (89% deteksi benar)
├─ Recall: 0.88 (88% kendaraan terdeteksi)
└─ F1: 0.88 (performa terbaik)

Jika pakai confidence 0.3:
├─ Recall tinggi (96%) tapi banyak false positive
├─ Precision rendah (82%)
└─ Banyak salah deteksi

Jika pakai confidence 0.9:
├─ Precision tinggi (98%) tapi ketinggalan banyak
├─ Recall rendah (45%)
└─ Banyak miss detections
```

---

## 4. IoU (Intersection over Union) ANALYSIS

Menunjukkan: **Seberapa akurat positioning bounding box?**

### Format Data:

```json
{
  "iou_thresholds": {
    "iou_0.5": {
      "precision": 0.920,
      "recall": 0.915,
      "description": "Box overlap >50% (standar)"
    },
    "iou_0.75": {
      "precision": 0.780,
      "recall": 0.721,
      "description": "Box overlap >75% (ketat)"
    },
    "iou_0.9": {
      "precision": 0.450,
      "recall": 0.380,
      "description": "Box overlap >90% (sangat ketat)"
    }
  }
}
```

Artinya:
- **IoU 0.5**: Box harus overlap >50% (sudah cukup)
- **IoU 0.75**: Box harus overlap >75% (lebih ketat)
- **IoU 0.9**: Box harus overlap >90% (sangat presisi)

---

## 5. VIDEO COUNTING DATA

Data hasil deteksi pada video.

### Format Data (JSON):

```json
{
  "video_analysis": {
    "video_path": "test_videos/traffic_short.mp4",
    "duration_seconds": 30,
    "fps": 15,
    "total_frames": 450,
    "processing_time_sec": 22.5,
    "fps_inference": 20.0,
    
    "detections": {
      "total_vehicles": 487,
      "mobil": 200,
      "bus": 150,
      "truk": 137,
      "confidence_avg": 0.89
    },
    
    "per_frame_stats": {
      "min_detections_per_frame": 0,
      "max_detections_per_frame": 8,
      "avg_detections_per_frame": 1.08,
      "frames_with_detections": 420,
      "frames_empty": 30
    },
    
    "confidence_distribution": {
      "0.8_0.9": 145,
      "0.9_0.95": 215,
      "0.95_1.0": 127
    }
  }
}
```

---

## 6. SAMPLE OUTPUT - LAPORAN LENGKAP

Contoh output yang sebenarnya dari `evaluate_model.py`:

### File: `evaluation_report.md`

```markdown
# YOLO Model Evaluation Report

## Date: 2026-02-13
## Model: yolov8n (runs/detect/train/weights/best.pt)
## Test Dataset: 250 images (37 test set)

---

## OVERALL PERFORMANCE

| Metric | Value |
|--------|-------|
| Precision | 0.892 |
| Recall | 0.915 |
| mAP50 | 0.867 |
| mAP50-95 | 0.713 |
| F1-Score | 0.903 |

**Interpretation**: Model sangat bagus untuk deteksi! ✓

---

## PER-CLASS BREAKDOWN

### Class: Mobil
- Precision: 0.895
- Recall: 0.945
- mAP50: 0.895
- True Positives: 1890
- False Positives: 230
- False Negatives: 110

### Class: Bus
- Precision: 0.932
- Recall: 0.895
- mAP50: 0.842
- True Positives: 1700
- False Positives: 125
- False Negatives: 200

### Class: Truk
- Precision: 0.942
- Recall: 0.895
- mAP50: 0.865
- True Positives: 1700
- False Positives: 105
- False Negatives: 200

---

## CONFIDENCE THRESHOLD ANALYSIS

Rekomendasi: Gunakan confidence = 0.5 (balance optimal)

### Threshold Analysis:

| Threshold | Precision | Recall | F1 | Count |
|-----------|-----------|--------|-------|-------|
| 0.3 | 0.81 | 0.96 | 0.88 | 5800 |
| 0.5 | 0.89 | 0.88 | 0.88 | 5200 |
| 0.7 | 0.94 | 0.75 | 0.83 | 4200 |

---

## CONFUSION MATRIX

```
                 Predicted
            Mobil    Bus    Truk
          ┌───────┬──────┬──────┐
Actual:   │       │      │      │
  Mobil   │ 1890  │ 60   │ 50   │
  Bus     │ 95    │ 1700 │ 105  │
  Truk    │ 70    │ 130  │ 1700 │
          └───────┴──────┴──────┘
```

---

## SUMMARY

✓ Model ready untuk production
✓ Semua class sudah >84% accuracy
✓ False positive rate <10%

Siap untuk deployment!
```

---

## 7. IMPLEMENTASI - PYTHON SCRIPT

Berikut script untuk generate semua data ini:

```python
# file: generate_accuracy_data.py

import json
import numpy as np
from ultralytics import YOLO

class AccuracyDataGenerator:
    def __init__(self, model_path, test_dir):
        self.model = YOLO(model_path)
        self.test_dir = test_dir
        self.classes = {0: 'mobil', 1: 'bus', 2: 'truk'}
        
    def generate_all_metrics(self):
        """Generate semua accuracy metrics"""
        
        # 1. Run validation
        results = self.model.val(data='data/data.yaml')
        
        # 2. Extract confusion matrix
        confusion_matrix = results.confusion_matrix.matrix
        
        # 3. Per-class metrics
        per_class = self._extract_per_class_metrics(results)
        
        # 4. Confidence threshold analysis
        threshold_analysis = self._analyze_confidence_thresholds()
        
        # Save all data
        output = {
            'overall_metrics': {
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            },
            'per_class_metrics': per_class,
            'confusion_matrix': confusion_matrix.tolist(),
            'confidence_analysis': threshold_analysis
        }
        
        # Save to JSON
        with open('accuracy_metrics.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    def _extract_per_class_metrics(self, results):
        """Extract metrics per class"""
        per_class = {}
        for i, class_name in self.classes.items():
            per_class[class_name] = {
                'precision': float(results.box.mp_per_class[i]),
                'recall': float(results.box.mr_per_class[i]),
                'mAP50': float(results.box.map50_per_class[i]),
                'mAP50_95': float(results.box.map_per_class[i])
            }
        return per_class
    
    def _analyze_confidence_thresholds(self):
        """Analyze different confidence thresholds"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        analysis = []
        
        for conf in thresholds:
            results = self.model.val(conf=conf)
            analysis.append({
                'confidence': conf,
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            })
        
        return analysis

# Usage:
if __name__ == "__main__":
    generator = AccuracyDataGenerator(
        model_path='runs/detect/train/weights/best.pt',
        test_dir='data/images/test'
    )
    
    metrics = generator.generate_all_metrics()
    
    print("=== OVERALL METRICS ===")
    print(json.dumps(metrics['overall_metrics'], indent=2))
    
    print("\n=== PER-CLASS METRICS ===")
    print(json.dumps(metrics['per_class_metrics'], indent=2))
```

---

## 8. CARA MENGGUNAKAN DATA INI UNTUK THESIS

Untuk bab **"Hasil & Pembahasan"** tulis:

```markdown
### 4.2 Hasil Evaluasi Model

#### 4.2.1 Overall Performance

Menggunakan metrik standar COCO:

Tabel 4.1: Overall Performance Metrics
| Metric | Value |
|--------|-------|
| Precision | 0.892 |
| Recall | 0.915 |
| mAP50 | 0.867 |
| mAP50-95 | 0.713 |

Hasil menunjukkan bahwa model mencapai precision 89.2% dan recall 91.5%, 
yang berarti model mampu mendeteksi 91.5% total kendaraan dengan 89.2% akurasi deteksi.

#### 4.2.2 Per-Class Analysis

Tabel 4.2: Per-Class Detection Performance
[Tabel confusion matrix]

Model memiliki performa terbaik pada class Truk (94.2% precision) 
dan terbaik pada recall untuk Mobil (94.5% recall).

#### 4.2.3 Confidence Threshold Optimization

Gambar 4.1: Precision-Recall Trade-off
[Grafik confidence analysis]

Threshold optimal = 0.5 (F1 score: 0.88)
```

---

## 9. CHECKLIST - ACCURACY DATA YANG HARUS ADA

```
✓ Confusion Matrix
✓ Per-class metrics (Precision, Recall, mAP)
✓ Confidence threshold analysis
✓ IoU analysis
✓ Video counting statistics
✓ Evaluation report (markdown)
✓ Visualizations (PNG):
  - confusion_matrix.png
  - pr_curve.png
  - confidence_analysis.png
✓ Raw JSON data untuk reference
```

---

## Summary

**Accuracy testing menghasilkan 6 tipe data:**

1. **Confusion Matrix** → Lihat kesalahan klasifikasi
2. **Per-Class Metrics** → Performance setiap class
3. **Confidence Analysis** → Optimal threshold
4. **IoU Analysis** → Bounding box precision
5. **Video Counting** → Deteksi pada video nyata
6. **Report** → Interpretasi lengkap

**Untuk thesis minimum perlu:**
- Confusion matrix (wajib)
- Per-class metrics (wajib)
- Confidence analysis (recommended)
- Report + interpretasi (wajib)

---

**Sudah siap? Mari implementasikan dengan data real dari model kamu!** 🚀
