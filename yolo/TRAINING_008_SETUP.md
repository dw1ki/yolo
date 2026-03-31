# YOLO Model Training Update - Video 008

**Status**: ✅ Konfigurasi selesai - siap untuk training

## Perubahan yang Telah Dilakukan

### 1. Model Lama (Backup)
- **Status**: ❌ Dihapus dari penggunaan aktif
- **Nama**: vehicle_night2
- **Lokasi Backup**: `models_backup/best_vehicle_night2_backup.pt`
- **File Size**: 6.2 MB

### 2. Dataset Baru (Video 008)
- **Sumber**: `input_videos/008.mp4`
- **Lokasi**: `data/dataset_008/`
- **Total Frames**: 641 training images
- **Training Split**: ~514 frames (80%)
- **Validation Split**: ~129 frames (20%)
- **Format**: YOLO Format (normalized bounding boxes)

### 3. File Konfigurasi yang Diupdate

#### `data.yaml` (Konfigurasi Dataset Utama)
```yaml
path: d:/backup/pktj/backend/yolo/data/dataset_008
train: images/train
val: images/val

nc: 3
names:
  0: Car
  1: Bus
  2: Truck
```

#### `train.py` (Script Training)
- Base Model: `yolov8m.pt`
- Epochs: 60
- Image Size: 1280
- Batch Size: 4
- Device: GPU (device=0)
- Output akan tersimpan di: `runs/train_video_008/`

## Timeline Perubahan

| Waktu | Aksi | Status |
|-------|------|--------|
| 2026-03-23 | Backup model vehicle_night2 | ✅ Selesai |
| 2026-03-23 | Update data.yaml ke dataset_008 | ✅ Selesai |
| 2026-03-23 | Update train.py dengan config optimal | ✅ Selesai |
| Pending | Jalankan training | ⏳ Siap |

## Cara Menggunakan

### Memulai Training
```bash
cd backend/yolo
python train.py
```

Training akan:
1. Load base model `yolov8m.pt`
2. Membaca dataset dari `dataset_008/`
3. Training selama 60 epochs
4. Menyimpan hasil di `runs/train_video_008/weights/best.pt`

### Hasil Training
Setelah selesai, model terbaik akan tersimpan di:
```
runs/train_video_008/weights/best.pt
```

### Menggunakan Model Baru
Untuk menggunakan model terlatih:
```python
from ultralytics import YOLO

model = YOLO("runs/train_video_008/weights/best.pt")
results = model.predict("path/to/video.mp4")
```

## Struktur Data

```
data/dataset_008/
├── images/
│   ├── train/          (514 images)
│   └── val/            (129 images)
└── labels/
    ├── train/          (514 label files)
    └── val/            (129 label files)

data_008.yaml           (Konfigurasi alternatif, sudah ada)
```

## Kelas Objek
- **Class 0**: Car (Mobil)
- **Class 1**: Bus
- **Class 2**: Truck (Truk)

## Catatan Penting
1. ✅ Dataset sudah ter-split 80:20 (train:val)
2. ✅ Semua label files sudah dalam format YOLO
3. ✅ Model base `yolov8m.pt` akan diunduh otomatis jika belum ada
4. ⚠️ Pastikan GPU tersedia untuk training (batch_size=4)
5. ⚠️ Training akan memakan waktu tergantung GPU (estimasi ~2-3 jam untuk yolov8m)

## Rollback (Jika Diperlukan)
Jika perlu kembali ke model lama:
```bash
copy models_backup\best_vehicle_night2_backup.pt models\best.pt
```

---
Created: 2026-03-23
Status: Siap untuk Training
Next Step: Jalankan `python train.py` untuk memulai
