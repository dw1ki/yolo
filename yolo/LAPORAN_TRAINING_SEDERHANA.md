# 📚 LAPORAN PELATIHAN MODEL YOLO - BAHASA SEDERHANA

**Dokumentasi Lengkap: Cara Melatih Model YOLOv8 untuk Deteksi Kendaraan**

---

## DAFTAR ISI
1. [Ringkasan Singkat](#ringkasan-singkat)
2. [Persiapan Data](#persiapan-data)
3. [Cara Melatih Model](#cara-melatih-model)
4. [Hasil yang Diharapkan](#hasil-yang-diharapkan)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Jika Ada Masalah](#jika-ada-masalah)

---

## RINGKASAN SINGKAT

### Apa itu Model YOLO?
Model YOLO adalah sebuah program yang bisa mengenali dan menghitung kendaraan dalam sebuah foto atau video.

### Yang Kami Gunakan:
- **Model**: YOLOv8 Medium (ukuran sedang, cukup akurat)
- **Tugas**: Mengenali 3 jenis kendaraan (Mobil, Bus, Truk)
- **Waktu Training**: 2-4 jam jika pakai GPU (kartu grafis)
- **Ukuran Foto**: 1280x1280 pixel

### Hasil yang Ingin Dicapai:
```
✓ Model bisa mengenali kendaraan dengan baik (>80% akurat)
✓ Bisa digunakan untuk analisis video
✓ Siap dipakai untuk penelitian/production
```

---

## PERSIAPAN DATA

### 1. Struktur Folder Data

Sebelum melatih, data harus diatur seperti ini:

```
backend/yolo/data/
├── images/              (folder foto)
│   ├── train/          (70% foto untuk melatih)
│   └── val/            (30% foto untuk testing)
└── labels/             (folder label/anotasi)
    ├── train/          (label untuk foto training)
    └── val/            (label untuk foto testing)
```

### 2. Format Label (Cara Memberi Tanda pada Foto)

Setiap foto yang akan dilatih harus ada file `.txt` yang berisi koordinat dan jenis kendaraan.

**Contoh file label (`image001.txt`):**
```
0 0.456 0.632 0.234 0.567
1 0.234 0.789 0.123 0.456
2 0.512 0.345 0.234 0.345
```

Penjelasan:
- Angka pertama (0, 1, 2) = jenis kendaraan:
  - `0` = Mobil
  - `1` = Bus
  - `2` = Truk
- Angka berikutnya = posisi kendaraan dalam foto (nilai 0-1, dimana 0 = kiri/atas, 1 = kanan/bawah)

### 3. File Konfigurasi (data.yaml)

File `data.yaml` berisi informasi tentang dataset:

```yaml
path: /mnt/localdisk1/vehicle-night-yolo    # Lokasi data
train: data/images/train                     # Lokasi foto training
val: data/images/val                         # Lokasi foto testing

names:                                       # Nama-nama kendaraan
  0: mobil
  1: bus
  2: truk
```

### 4. Berapa Banyak Data yang Dibutuhkan?

```
🎯 Target Data:
├── Total Foto: 1000-2000 foto
├── Foto Training: ~1400 foto (70%)
├── Foto Testing: ~600 foto (30%)
└── Jumlah Kendaraan yang Ditandai: 3000-5000+ 

📊 Pembagian per Jenis:
├── Mobil: ~40-50% (paling banyak)
├── Bus: ~20-30%
└── Truk: ~20-30%
```

---

## CARA MELATIH MODEL

### Langkah 1: Persiapan (10-30 menit)

**A. Install program yang dibutuhkan:**
```bash
pip install ultralytics opencv-python PyYAML
```

**B. Periksa data sudah benar:**
```bash
python
>>> from pathlib import Path
>>> print("Foto training:", len(list(Path("data/images/train").glob("*.jpg"))))
>>> print("Foto testing:", len(list(Path("data/images/val").glob("*.jpg"))))
```

### Langkah 2: Jalankan Training (2-4 jam)

Buat file bernama `train.py` dengan isi:

```python
from ultralytics import YOLO

# Muat model dasar (program akan otomatis unduh)
model = YOLO("yolov8n.pt")

# Mulai training dengan konfigurasi
model.train(
    data="data.yaml",          # File konfigurasi data
    epochs=60,                 # Berapa kali melihat semua data
    imgsz=1280,                # Ukuran foto (pixel)
    batch=4,                   # Berapa foto diproses sekaligus
    device=0,                  # Gunakan GPU (kartu grafis)
    workers=2,                 # Berapa thread untuk loading data
    
    # Penggandaan data (agar model lebih robust)
    hsv_v=0.4,                 # Ubah kecerahan secara acak
    hsv_s=0.4,                 # Ubah warna secara acak
    mosaic=0.2,                # Gabung 4 foto jadi 1
    
    # Berhenti jika tidak ada improvement
    patience=20,               # Tunggu 20 epoch jika tidak improve
)
```

Terus jalankan:
```bash
cd backend/yolo
python train.py
```

### Langkah 3: Proses Training (Otomatis)

Selama training, kami akan lihat:

```
Epoch 1/60
 100%|██████████| 175/175 [00:45<00:00,  3.87it/s]
          Class     Images     Targets           P           R      mAP50   mAP50-95
      mobil       300       450       0.856       0.892       0.875       0.654
        bus       300       120       0.745       0.823       0.784       0.512
       truk       300       180       0.812       0.876       0.844       0.598
       all       300       750       0.804       0.863       0.834       0.588

Validating...
Computing metrics...
```

Apa yang terjadi di sini:
- P (Precision) = Dari kendaraan yang terdeteksi, berapa % yang benar
- R (Recall) = Dari semua kendaraan, berapa % yang terdeteksi
- mAP50 = Skor akurasi keseluruhan (target > 0.80)

### Langkah 4: Hasil Training (Otomatis Tersimpan)

Setelah selesai, model terbaik disimpan di:

```
runs/detect/train/
├── weights/
│   ├── best.pt              ← Model terbaik (GUNAKAN INI!)
│   └── last.pt              ← Model terakhir
├── results.png              ← Grafik training
├── confusion_matrix.png     ← Matrix kesalahan
├── F1_curve.png            ← Kurva F1-Score
└── PR_curve.png            ← Kurva Precision-Recall
```

---

## HASIL YANG DIHARAPKAN

### Penjelasan Metrics (Ukuran Akurasi)

#### 1. **Precision (P)** - Keakuratan Deteksi
```
Arti: Dari 100 kendaraan yang model deteksi, berapa banyak yang BENAR?

Contoh:
  Model mengatakan: "Ada 100 mobil"
  Kenyataannya: "80 mobil benar, 20 salah"
  Precision = 80/100 = 0.80 (80%)

Target: > 0.75 (lebih dari 75%)
```

#### 2. **Recall (R)** - Cakupan Deteksi
```
Arti: Dari semua kendaraan yang ADA, berapa % yang berhasil TERDETEKSI?

Contoh:
  Di video ada: 100 mobil
  Yang terdeteksi: 85 mobil
  Recall = 85/100 = 0.85 (85%)

Target: > 0.80 (lebih dari 80%)
```

#### 3. **mAP50** - Skor Keseluruhan
```
Arti: Rata-rata akurasi keseluruhan model

Target: > 0.80 (lebih dari 80%)
Gunakan untuk: Penilaian keseluruhan model
```

#### 4. **F1-Score** - Keseimbangan
```
Arti: Mixing antara Precision dan Recall

Target: > 0.78 (lebih dari 78%)
Gunakan untuk: Keseimbangan antara akurasi dan cakupan
```

### Hasil yang Realistis

Jika data bagus dan training berhasil:

```
✅ HASIL BAIK:
├─ Precision: 0.82-0.88 (82-88%)
├─ Recall: 0.85-0.92 (85-92%)
├─ mAP50: 0.83-0.89 (83-89%)
└─ F1-Score: 0.84-0.90 (84-90%)

Artinya: Model siap pakai ✓
```

Jika hasil rendah:

```
⚠️ HASIL KURANG BAIK:
├─ Precision/Recall < 0.70
├─ mAP50 < 0.75
└─ Ada masalah yang perlu diperbaiki

Kemungkinan masalah:
├─ Data terlalu sedikit
├─ Kualitas label jelek (label salah)
├─ Hyperparameter tidak cocok
└─ Model terlalu kecil
```

---

## CARA MENJALANKAN

### Opsi 1: Training dari Awal

```bash
cd backend/yolo
python train.py
```

Training akan:
- Otomatis download model dasar (hanya sekali)
- Baca file data.yaml
- Muat foto training dan testing
- Mulai training selama 60 epoch
- Simpan model terbaik ke `runs/detect/train/weights/best.pt`

### Opsi 2: Lanjut Training (Jika Terputus)

Jika training tiba-tiba berhenti, bisa dilanjutkan:

```bash
python -c "
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/last.pt')
model.train(
    data='data.yaml',
    epochs=60,
    resume=True
)
"
```

### Opsi 3: Gunakan Model untuk Prediksi

Setelah training selesai, pakai model:

```python
from ultralytics import YOLO

# Muat model yang sudah dilatih
model = YOLO("runs/detect/train/weights/best.pt")

# Deteksi pada gambar
results = model.predict(source="foto.jpg", conf=0.5)

# Deteksi pada video
results = model.predict(source="video.mp4", conf=0.5)

# Lihat hasil
for result in results:
    print(f"Ditemukan {len(result.boxes)} kendaraan")
```

---

## JIKA ADA MASALAH

### Masalah 1: "Error: CUDA out of memory"

**Gejala**: Program berhenti dengan error kehabisan memori GPU

**Solusi**:
- Turunkan batch size: `batch=2` (dari 4)
- Atau turunkan ukuran foto: `imgsz=640` (dari 1280)
- Model yang kami gunakan: `yolov8n.pt` (nano, paling ringan)

### Masalah 2: "Training sangat lambat"

**Gejala**: 1 epoch = 5+ menit (terlalu lama)

**Solusi**:
- Pastikan menggunakan GPU, bukan CPU
- Turunkan jumlah workers: `workers=0`
- Gunakan model lebih kecil: `yolov8n.pt`

### Masalah 3: "Akurasi tidak naik-naik"

**Gejala**: Setelah 20 epoch, metric tidak improve lagi

**Solusi**:
- Data mungkin terlalu sedikit (butuh minimal 500 per jenis)
- Tambah data baru
- Tambah penggandaan data (augmentation)
- Tingkatkan patience: `patience=50`

### Masalah 4: "Model bagus di training tapi jelek di real video"

**Gejala**: Accuracy tinggi tapi di video nyata buruk

**Solusi**:
- Kumpulkan lebih banyak data yang beragam
- Tambah augmentation (penggandaan data)
- Fine-tune dengan data dari dunia nyata

---

## PENJELASAN SEDERHANA SETIAP PARAMETER

| Parameter | Nilai | Penjelasan Sederhana |
|-----------|-------|---------------------|
| **epochs** | 60 | Berapa kali model melihat seluruh data training |
| **imgsz** | 1280 | Ukuran foto (pixel). Lebih besar = lebih detail tapi lebih lambat |
| **batch** | 4 | Berapa foto diproses sekaligus. Lebih besar = lebih cepat tapi butuh GPU lebih besar |
| **device** | 0 | Gunakan GPU pertama (0=GPU, 'cpu'=CPU) |
| **workers** | 2 | Berapa proses loading data. Lebih banyak = lebih cepat (jika CPU bagus) |
| **hsv_v** | 0.4 | Ubah kecerahan foto secara acak (untuk kondisi siang/malam) |
| **hsv_s** | 0.4 | Ubah warna foto secara acak |
| **mosaic** | 0.2 | 20% waktu, gabung 4 foto jadi 1 (meningkatkan variasi) |
| **patience** | 20 | Jika 20 epoch tidak ada improvement, berhenti training |

---

## CHECKLIST SEBELUM TRAINING

```
✓ Folder data sudah siap
  - data/images/train/ berisi foto
  - data/images/val/ berisi foto
  - data/labels/train/ berisi .txt files
  - data/labels/val/ berisi .txt files

✓ File data.yaml sudah benar
  - Path menunjuk ke data/ yang tepat
  - Classes: mobil, bus, truk

✓ Python libraries sudah terinstall
  - pip install ultralytics opencv-python

✓ GPU siap pakai (jika ada)
  - Device 0 = GPU pertama
```

---

## TIPS & TRIK

### 1. Cek Progress Training
Saat training berjalan, Anda bisa lihat progress di console. Jika lambat, bisa dihentikan dengan `Ctrl+C` dan cek masalahnya.

### 2. Lihat Hasil Training
Setelah training selesai, buka folder `runs/detect/train/` untuk lihat:
- `results.png` - Grafik performa selama training
- `confusion_matrix.png` - Matrix kesalahan klasifikasi
- `best.pt` - Model terbaik (gunakan ini!)

### 3. Model Terbaik vs Terakhir
- `best.pt` = Model dengan akurasi tertinggi (GUNAKAN INI)
- `last.pt` = Model di epoch terakhir (bisa jelek)

### 4. Mulai dari Model Sudah Dilatih
Jika Anda ingin fine-tune dari model yang sudah ada:
```python
# Gunakan `yolov8n.pt` yang sudah kami setup
model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data.yaml", epochs=20)  # Training lebih sedikit
```

---

## RINGKASAN

**Proses Training Model YOLO:**

1. **Persiapkan data** → Foto + label dalam struktur yang benar
2. **Konfigurasi** → Atur data.yaml dan parameter training
3. **Jalankan training** → `python train.py`
4. **Monitor** → Lihat metric P, R, mAP50 meningkat
5. **Selesai** → Model terbaik di `runs/detect/train/weights/best.pt`
6. **Gunakan** → Pakai model untuk deteksi foto/video

**Waktu yang dibutuhkan:**
- Persiapan data: 1-2 jam
- Training: 2-4 jam (dengan GPU)
- Testing & evaluasi: 1-2 jam

---

**Dibuat**: 13 Februari 2026  
**Status**: ✅ Siap Digunakan  
**Untuk**: Tugas Akhir Mahasiswa - Deteksi Kendaraan dengan YOLO
