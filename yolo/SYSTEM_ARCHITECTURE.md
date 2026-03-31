# 📐 SYSTEM ARCHITECTURE DIAGRAM - YOLO EVALUATION SYSTEM

## COMPONENT DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    YOLO MODEL EVALUATION SYSTEM                           ║
║                          Architecture Overview                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT SOURCES                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │   Training   │   │   Validation │   │    Test      │              │
│  │   Dataset    │   │    Dataset   │   │    Video     │              │
│  │              │   │              │   │              │              │
│  │ data/images/ │   │ data/images/ │   │  test.mp4    │              │
│  │    train/    │   │      val/    │   │              │              │
│  └──────────────┘   └──────────────┘   └──────────────┘              │
│         │                  │                   │                       │
└─────────────────────────────────────────────────────────────────────────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────┐
        │    YOLO Model Loading           │
        │  (runs/detect/train/weights/    │
        │   best.pt or yolov8n.pt)        │
        └─────────────────────────────────┘
                            │
                            ▼
      ┌──────────────────────────────────────────┐
      │     YOLOEvaluator Core Engine            │
      │  ┌────────────────────────────────────┐  │
      │  │ Phase 1: YOLO Built-in Validation  │  │
      │  │  - model.val()                     │  │
      │  │  - Metrics: P, R, mAP50, mAP50-95  │  │
      │  └────────────────────────────────────┘  │
      │  ┌────────────────────────────────────┐  │
      │  │ Phase 2: Custom Test Dataset       │  │
      │  │  - Load GT annotations (YOLO fmt) │  │
      │  │  - IoU matching                    │  │
      │  │  - Per-class metrics               │  │
      │  └────────────────────────────────────┘  │
      │  ┌────────────────────────────────────┐  │
      │  │ Phase 3: Video Analysis            │  │
      │  │  - Frame-by-frame inference        │  │
      │  │  - Vehicle counting & tracking     │  │
      │  │  - Annotated output video          │  │
      │  └────────────────────────────────────┘  │
      │  ┌────────────────────────────────────┐  │
      │  │ Phase 4: Advanced Analysis         │  │
      │  │  - Confusion matrix                │  │
      │  │  - Confidence threshold tuning     │  │
      │  │  - Precision-Recall curves         │  │
      │  └────────────────────────────────────┘  │
      └──────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    ┌─────────────┐  ┌────────────────┐  ┌────────────┐
    │   Metrics   │  │  Visualizations│  │   Report   │
    │ Calculation │  │    & Plots     │  │  Generation│
    │             │  │                │  │            │
    │ - Precision │  │ - Confusion    │  │ - Markdown │
    │ - Recall    │  │   Matrix       │  │   Report   │
    │ - mAP50     │  │ - PR Curve     │  │ - JSON     │
    │ - F1-Score  │  │ - Threshold    │  │   Export   │
    │             │  │   Analysis     │  │            │
    └─────────────┘  └────────────────┘  └────────────┘
           │                ▼                ▼
           │         ┌────────────────┐  ┌────────────┐
           │         │ confusion_     │  │evaluation_ │
           │         │ matrix.png     │  │ report.md  │
           │         └────────────────┘  └────────────┘
           │
           └──────────────┬───────────────┘
                          ▼
           ┌──────────────────────────────┐
           │   OUTPUT & VISUALIZATION     │
           ├──────────────────────────────┤
           │ ✓ confusion_matrix.png       │
           │ ✓ confidence_analysis.png    │
           │ ✓ pr_curve.png               │
           │ ✓ evaluation_report.md       │
           │ ✓ output_annotated.mp4       │
           └──────────────────────────────┘
```

---

## CLASS DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          YOLOEvaluator Class                              ║
║                          Object-Oriented Design                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│                       YOLOEvaluator                                  │
├──────────────────────────────────────────────────────────────────────┤
│ ATTRIBUTES:                                                          │
│  - model: YOLO                          [Loaded YOLO model]         │
│  - data_yaml: str                       [Config file path]          │
│  - classes: Dict[int, str]              [Class mapping]             │
│  - results: Dict                        [Evaluation results]        │
├──────────────────────────────────────────────────────────────────────┤
│ METHODS (Core Evaluation):                                           │
│                                                                      │
│  + validate_with_yolo()                                             │
│    ├─ Run built-in YOLO validation                                  │
│    ├─ Return: Dict[metrics]                                         │
│    └─ Output: Precision, Recall, mAP50, mAP50-95                   │
│                                                                      │
│  + evaluate_test_dataset(test_dir: str)                             │
│    ├─ Evaluate on custom test images                                │
│    ├─ Load GT annotations (YOLO format)                             │
│    ├─ Match predictions to ground truth                             │
│    └─ Return: Dict[metrics, per_class]                              │
│                                                                      │
│  + evaluate_video(video_path, output_video)                         │
│    ├─ Analyze video frame-by-frame                                  │
│    ├─ Count vehicles per class                                      │
│    ├─ Generate annotated output                                     │
│    └─ Return: Dict[video_stats]                                     │
│                                                                      │
│  + generate_confusion_matrix(test_dir)                              │
│    ├─ Create per-class confusion matrix                             │
│    ├─ Visualize as heatmap                                          │
│    └─ Identify misclassifications                                   │
│                                                                      │
│  + analyze_confidence_threshold(test_dir)                           │
│    ├─ Test multiple confidence values                               │
│    ├─ Find optimal threshold                                        │
│    └─ Return: Dict[threshold -> metrics]                            │
├──────────────────────────────────────────────────────────────────────┤
│ METHODS (Utility):                                                   │
│                                                                      │
│  - _load_classes()                      [Load from data.yaml]       │
│  - _load_yolo_annotations(file)         [Parse .txt labels]        │
│  - _extract_predictions(result)         [Extract from YOLO]        │
│  - _match_boxes(pred, gt)               [IoU-based matching]       │
│  - _calculate_iou(box1, box2)           [Intersection over Union]   │
│  - _compute_metrics(tp, fp, fn)         [P, R, F1, mAP calc]       │
│                                                                      │
│ METHODS (Reporting):                                                 │
│                                                                      │
│  - _print_metrics(metrics)              [Console output]            │
│  - _print_per_class_metrics(per_class)  [Per-class stats]          │
│  - _plot_confusion_matrix(matrix)       [Heatmap visualization]    │
│  - _plot_confidence_analysis(results)   [Threshold analysis]       │
│  - generate_report(output_file)         [Markdown report]           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## DATA FLOW DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    EVALUATION PROCESS - DATA FLOW                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

                        START EVALUATION
                              │
                              ▼
                    ┌──────────────────┐
                    │ Load YOLO Model  │
                    └─────────┬────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
   ╔════════════▼══════════╗  │  ╔═════════▼═══════════╗
   ║ PHASE 1: VALIDATION  ║  │  ║ PHASE 2: TEST DATA  ║
   ║ Built-in YOLO        ║  │  ║ Custom Evaluation   ║
   ║                      ║  │  ║                     ║
   ║ data/images/val/ ────┼──┼─→ data/images/test/   ║
   ║         │            ║  │  ║         │           ║
   ║         ▼            ║  │  ║         ▼           ║
   ║ model.val()          ║  │  ║ For each image:     ║
   ║         │            ║  │  ║  - Load GT labels   ║
   ║         ▼            ║  │  ║  - Run inference    ║
   ║ Get metrics          ║  │  ║  - Match boxes      ║
   ║ (P,R,mAP50,mAP50-95) ║  │  ║  - Calculate IoU    ║
   ║         │            ║  │  ║  - Update metrics   ║
   ║         ▼            ║  │  ║         │           ║
   ║ results['metrics']   ║  │  ║         ▼           ║
   ╚════════════┬═════════╝  │  ║ TP, FP, FN, per_    ║
               │            │  ║ class counts        ║
               │            │  ║         │           ║
               │            │  ║         ▼           ║
               │            │  ║ results['per_class']║
               │            │  ╚═════════┬═════════╝
               │            │            │
               │    ┌───────┴────────────┘
               │    │
               │    ▼
   ╔════════════════════════════╗
   ║ PHASE 3: VIDEO ANALYSIS    ║
   ║                            ║
   ║ test.mp4 ──────────────┐   ║
   ║                        ▼   ║
   ║                   Loop frames:
   ║                        │   ║
   ║                        ▼   ║
   ║              Run inference │
   ║                        │   ║
   ║                        ▼   ║
   ║          Count vehicles    ║
   ║          by class          ║
   ║                        │   ║
   ║                        ▼   ║
   ║         Write to output    ║
   ║         video with boxes   ║
   ║                        │   ║
   ║                        ▼   ║
   ║      results['video_stats']║
   ╚─────────────┬──────────────╝
                │
                ▼
   ╔═════════════════════════════════╗
   ║ PHASE 4: ADVANCED ANALYSIS      ║
   ║                                 ║
   ║ 4a. Confusion Matrix:           ║
   ║     Actual vs Predicted classes ║
   ║                                 ║
   ║ 4b. Confidence Threshold:       ║
   ║     Test 0.3 - 0.9             ║
   ║     Find optimal balance        ║
   ║                                 ║
   ║ 4c. Per-class Analysis:         ║
   ║     Mobil, Bus, Truk metrics   ║
   ╚─────────────┬───────────────────┘
                │
   ┌────────────┴─────────────┐
   │                          │
   ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│ VISUALIZATION   │    │ REPORT GENERATION    │
├─────────────────┤    ├──────────────────────┤
│ PNG Plots:      │    │ Markdown Report:     │
│ - Confusion mtx │    │ - Summary metrics    │
│ - PR curve      │    │ - Per-class summary  │
│ - F1 vs thresh  │    │ - Recommendations    │
│                 │    │                      │
│ Annotated Video:│    │ JSON Export:         │
│ - With boxes    │    │ - Machine readable   │
│ - With labels   │    │ - For dashboards     │
└─────────────────┘    └──────────────────────┘
   │                        │
   └────────────┬───────────┘
                ▼
        ┌────────────────────┐
        │  FINAL OUTPUTS     │
        ├────────────────────┤
        │ ✓ confusion_       │
        │   matrix.png       │
        │ ✓ confidence_      │
        │   analysis.png     │
        │ ✓ evaluation_      │
        │   report.md        │
        │ ✓ output_          │
        │   annotated.mp4    │
        └────────────────────┘
                │
                ▼
           USER REVIEW
                │
                ▼
        DEPLOY / RETRAIN
```

---

## SEQUENCE DIAGRAM - Evaluation Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║          EVALUATION PROCESS - SEQUENCE DIAGRAM                       ║
╚══════════════════════════════════════════════════════════════════════╝

User              Evaluator          YOLO Model       Filesystem
 │                   │                   │                │
 ├─ Run Script ─────→│                   │                │
 │                   │                   │                │
 │         ┌─────────┴─────────┐        │                │
 │         │ Load Model        │        │                │
 │         │ & Config          │        │                │
 │         └─────────┬─────────┘        │                │
 │                   │                   │                │
 │                   │ model.val() ─────→│                │
 │                   │                   │ Process val/   │
 │                   │ mAP50, Precision ←┤ Compute       │
 │                   │                   │ metrics       │
 │                   │                   │                │
 │         ┌─────────┴─────────┐        │                │
 │         │ Load Test Images  │        │                │
 │         │ & GT Labels       │        │                │
 │         └─────────┬─────────┘        │                │
 │                   │                   │                │
 │     For Each Image:                  │                │
 │         │                            │                │
 │         ├─ Load Image ──────────────────────────────→│
 │         │  Load GT Labels ────────────────────────→│
 │         │  Run Prediction ───────────→│            │
 │         │  Extract boxes ────────────←│            │
 │         │  Match Boxes (IoU)          │            │
 │         │  Calculate TP/FP/FN         │            │
 │         │  Update Metrics             │            │
 │         │                             │            │
 │     [Loop All Images]                 │            │
 │         │                             │            │
 │         ├─ Compute Final Metrics      │            │
 │         │  (Precision, Recall,F1)     │            │
 │         │                             │            │
 │         ├─ Video Analysis             │            │
 │         │  ├─ Open Video              │            │
 │         │  ├─ Loop Frames             │            │
 │         │  │  ├─ Predict ────────────→│            │
 │         │  │  ├─ Count Vehicles  ←────┤            │
 │         │  │  └─ Write Output         │            │
 │         │  ├─ Save Annotated Video ───────────────→│
 │         │  └─ Generate Stats  ←────────            │
 │         │                             │            │
 │         ├─ Generate Plots             │            │
 │         │  ├─ Confusion Matrix ──────────────────→│
 │         │  ├─ PR Curve           ────────────────→│
 │         │  └─ Threshold Analysis ───────────────→│
 │         │                             │            │
 │         ├─ Generate Report ──────────────────────→│
 │         │                             │            │
 │  ←──────┤ Return Results              │            │
 │         │                             │            │
 └─ Display Summary ─────────────────────────────────→│
           │                             │            │
      ✅ COMPLETE                        │            │
```

---

## METRICS CALCULATION FLOW

```
╔═══════════════════════════════════════════════════════════════════╗
║           HOW METRICS ARE CALCULATED - DETAILED FLOW             ║
╚═══════════════════════════════════════════════════════════════════╝

1. BOX MATCHING (Using IoU)
   ─────────────────────────
   
   Ground Truth Box        Predicted Box
        │                       │
        └─────────┬─────────────┘
                  │
        Calculate Intersection over Union
                  │
        IoU = Area(Intersection) / Area(Union)
                  │
        ┌─────────┴────────┐
        │                  │
        ▼                  ▼
    IoU >= 0.5         IoU < 0.5
        │                  │
        ▼                  ▼
      MATCH              NO MATCH
        │                  │
        ▼                  ▼
      TP                 FP (if pred)
                         FN (if GT unmatched)


2. METRICS AGGREGATION
   ─────────────────────

   After processing all images:
   
   Total TP (True Positives)
   Total FP (False Positives)
   Total FN (False Negatives)
           │
           ├─ Precision = TP / (TP + FP)
           │
           ├─ Recall = TP / (TP + FN)
           │
           ├─ F1-Score = 2 * (P * R) / (P + R)
           │
           └─ Accuracy = TP / (TP + FP + FN)


3. mAP CALCULATION
   ────────────────

   For each class and confidence threshold:
   
   ├─ Sort predictions by confidence
   ├─ For each threshold:
   │  ├─ Calculate Precision at this point
   │  └─ Calculate Recall at this point
   │
   ├─ Draw Precision-Recall Curve
   ├─ Calculate Area Under Curve (AUC) = AP
   │
   └─ Average AP across all classes = mAP


4. PER-CLASS METRICS
   ──────────────────

   For each class (Mobil, Bus, Truk):
   
   Count TP, FP, FN for THIS CLASS ONLY
           │
           ├─ Class Precision
           ├─ Class Recall
           ├─ Class F1-Score
           └─ Class mAP


5. CONFUSION MATRIX
   ────────────────

   Actual\Predicted   Mobil   Bus  Truk
   ─────────────────────────────────────
   Mobil              450     20    30
   Bus                 15    120    25
   Truk                25     20   180
   
   From this:
   ├─ Recognize which classes are confused
   ├─ Identify if Bus↔Truk often confused
   └─ Plan improvements (more training data, etc)
```

---

## SYSTEM QUALITY METRICS

```
╔═════════════════════════════════════════════════════════════╗
║        QUALITY LEVELS - SCORING SYSTEM                      ║
╚═════════════════════════════════════════════════════════════╝

OVERALL QUALITY SCORE (0-100):

   Score   │ Status  │ Interpretation
   ────────┼─────────┼─────────────────────────────────────
   90-100  │ 🟢 A+   │ Production Ready - Deploy Now!
   80-89   │ 🟢 A    │ Ready - Monitor Closely
   70-79   │ 🟡 B    │ Acceptable - Consider Improvements
   60-69   │ 🟠 C    │ Concerning - Needs Retraining
   0-59    │ 🔴 F    │ Poor - Restart Training

Calculation:
   Score = (P * 0.25) + (R * 0.25) + (F1 * 0.25) + (mAP50 * 0.25)
           * 100

Example:
   P=0.85, R=0.88, F1=0.865, mAP50=0.88
   Score = (0.85 + 0.88 + 0.865 + 0.88) * 25 = 86.75 → Grade A
```

---

## DEPLOYMENT DECISION MATRIX

```
╔═════════════════════════════════════════════════════════════╗
║    SHOULD WE DEPLOY? - DECISION MATRIX                     ║
╚═════════════════════════════════════════════════════════════╝

Metric          │ Threshold | Status
────────────────┼───────────┼────────────
Precision (P)   │ > 0.75    │ Required
Recall (R)      │ > 0.80    │ Required
F1-Score        │ > 0.78    │ Required
mAP50           │ > 0.80    │ Required
mAP50-95        │ > 0.55    │ Nice-to-have
────────────────┼───────────┼────────────
Inference Speed │ < 0.1s    │ Required (per image)
Memory Usage    │ < 2GB     │ Required
────────────────┼───────────┼────────────

DECISION LOGIC:

   IF all_required_pass THEN
       ✅ DEPLOY TO PRODUCTION
   ELIF some_fail AND not_critical THEN
       ⚠️  DEPLOY WITH MONITORING
   ELSE
       ❌ RETRAIN - DO NOT DEPLOY
```

---

## FILES & OUTPUTS STRUCTURE

```
backend/yolo/
├── evaluate_model.py              ← Main evaluation script
├── TRAINING_REPORT.md             ← Training documentation
├── EVALUATION_GUIDE.md            ← This guide
├── SYSTEM_ARCHITECTURE.md         ← UML & diagrams
│
├── data.yaml                      ← Dataset config
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── runs/detect/train/
│   └── weights/
│       ├── best.pt               ← Best model
│       └── last.pt               ← Last checkpoint
│
├── test.mp4                       ← Test video
│
└── [EVALUATION OUTPUTS]
    ├── confusion_matrix.png       ← Heatmap
    ├── confidence_analysis.png    ← Threshold analysis
    ├── evaluation_report.md       ← Text report
    └── output_annotated.mp4       ← Video with detections
```

---

**Generated**: February 13, 2026  
**Version**: 1.0  
**Status**: Complete & Ready for Implementation
