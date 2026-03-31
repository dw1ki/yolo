# 📊 PANDUAN TEST AKURASI MODEL YOLO - BAHASA MUDAH

**Cara Menguji Seberapa Baik Model Mengenali Kendaraan**

---

## DAFTAR ISI
1. [Mulai Cepat](#mulai-cepat)
2. [Cara Testing Lengkap](#cara-testing-lengkap)
3. [Mengerti Hasil Test](#mengerti-hasil-test)
4. [Kalau Hasil Jelek](#kalau-hasil-jelek)
5. [Bersiap Pakai Produksi](#bersiap-pakai-produksi)

---

## MULAI CEPAT ⚡

### Opsi 1: Test dengan 1 Perintah (paling mudah)

Jika sudah punya `yolov8n.pt` atau `best.pt`:

```bash
cd backend/yolo

python -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.val(data='data.yaml', imgsz=1280, batch=4)

print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
print(f'mAP50: {results.box.map50:.4f}')
"
```

**Hasil yang akan keluar:**
```
Precision: 0.8456 (84.56%)
Recall: 0.8924 (89.24%)
mAP50: 0.8654 (86.54%)
```

### Opsi 2: Test pada Gambar Tunggal

```bash
python -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Deteksi gambar
results = model.predict(source='foto_test.jpg', conf=0.5)

# Lihat berapa kendaraan ditemukan
for result in results:
    print(f'Ditemukan {len(result.boxes)} kendaraan')
    for box in result.boxes:
        vehicle_class = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        print(f'  - {vehicle_class}: {confidence:.2f}')
"
```

---

## CARA TESTING LENGKAP

### Step 1: Jalankan Script Evaluasi

Kami sudah buatkan script otomatis bernama `evaluate_model.py`. Jalankan:

```bash
cd backend/yolo
python evaluate_model.py
```

Script ini akan secara otomatis:
1. Test model pada validation dataset
2. Hitung akurasi (Precision, Recall, mAP50, F1-Score)
3. Analisis video jika ada
4. Buat confusion matrix (matrix kesalahan)
5. Buat laporan

### Step 2: Lihat Hasil di Console

Output akan terlihat seperti ini:

```
============================================================
🧪 TEST AKURASI MODEL
============================================================

📊 HASIL KESELURUHAN
--------------------------------------------------
Precision:    0.8456 (84.56%)
Recall:       0.8924 (89.24%)
mAP50:        0.8654 (86.54%)
mAP50-95:     0.5432 (54.32%)

📊 HASIL PER JENIS KENDARAAN
--------------------------------------------------

Mobil:
  TP: 450, FP: 32, FN: 28
  Precision: 0.8821 (88.21%)
  Recall:    0.9412 (94.12%)
  F1-Score:  0.9109 (91.09%)

Bus:
  TP: 120, FP: 18, FN: 15
  Precision: 0.8696 (86.96%)
  Recall:    0.8889 (88.89%)
  F1-Score:  0.8791 (87.91%)

Truk:
  TP: 180, FP: 22, FN: 20
  Precision: 0.8910 (89.10%)
  Recall:    0.9000 (90.00%)
  F1-Score:  0.8955 (89.55%)
```

### Step 3: Lihat Visualisasi (Gambar)

Script juga membuat gambar-gambar:
- `confusion_matrix.png` - Matrix kesalahan
- `confidence_analysis.png` - Analisis keyakinan model
- `evaluation_report.md` - Laporan text

Buka folder `backend/yolo/` untuk lihat gambar-gambar tersebut.

---

## MENGERTI HASIL TEST

### Angka-Angka yang Penting

| Istilah | Arti | Target | Penjelasan |
|---------|------|--------|-----------|
| **Precision** | Keakuratan deteksi | > 0.75 | Dari kendaraan yang terdeteksi, berapa % yang benar? |
| **Recall** | Cakupan terdeteksi | > 0.80 | Dari semua kendaraan, berapa % yang terdeteksi? |
| **F1-Score** | Keseimbangan | > 0.78 | Campuran antara Precision dan Recall |
| **mAP50** | Skor akurasi | > 0.80 | Skor keseluruhan pada IoU 0.5 |
| **mAP50-95** | Skor ketat | > 0.55 | Skor keseluruhan yang lebih ketat |

### Contoh 1: "Hasil Sangat Bagus"

```
Precision: 0.87 ✓
Recall: 0.89 ✓
mAP50: 0.88 ✓
F1-Score: 0.88 ✓

Kesimpulan: Model siap dipakai! Akurat dan cakupan baik.
```

### Contoh 2: "Precision Tinggi, Recall Rendah"

```
Precision: 0.92 ✓ (Akurat)
Recall: 0.65 ✗ (Banyak terlewat)

Artinya: Kendaraan yang terdeteksi memang benar, tapi banyak yang hilang.

Penyebab: Model terlalu hati-hati, takut salah.

Solusi: Turunkan confidence threshold
>>> model.predict(source='video.mp4', conf=0.3)  # dari 0.5 ke 0.3
```

### Contoh 3: "Precision Rendah, Recall Tinggi"

```
Precision: 0.65 ✗ (Banyak kesalahan)
Recall: 0.92 ✓ (Menangkap semua)

Artinya: Model deteksi banyak, tapi banyak kesalahan.

Penyebab: Model terlalu berani, mendeteksi yang bukan kendaraan.

Solusi: Naikkan confidence threshold
>>> model.predict(source='video.mp4', conf=0.7)  # dari 0.5 ke 0.7
```

### Contoh 4: "Semua Metrik Rendah"

```
Precision: 0.52 ✗
Recall: 0.58 ✗
mAP50: 0.48 ✗

Artinya: Model tidak bekerja dengan baik.

Kemungkinan penyebab:
1. Data training terlalu sedikit (< 300 per jenis)
2. Kualitas label jelek (anotasi salah)
3. Jenis kendaraan sulit dibedakan
4. Model terlalu kecil untuk data besar
5. Hyperparameter tidak cocok

Solusi:
- Kumpulkan lebih banyak data
- Cek kualitas label/anotasi
- Kami gunakan model nano (yolov8n) untuk speed optimal
- Edit hyperparameter di train.py
```

### Penjelasan Matrix Kesalahan (Confusion Matrix)

```
              Diprediksi Model
              Mobil  Bus  Truk
Actual Mobil   450    20   30
       Bus      15   120   25
       Truk     25    20  180

Arti:
- Diagonal (gelap): Prediksi benar (450 mobil benar, 120 bus benar, dll)
- Off-diagonal (terang): Kesalahan
  - Contoh: 20 bus diprediksi sebagai mobil

Pelajaran:
- Jika ada baris/kolom gelap = model bagus untuk jenis itu
- Jika banyak warna terang = sering salah kira
- Contoh: Jika Bus sering dikira Truk → Perlu data Bus lebih banyak
```

---

## KALAU HASIL JELEK

### Situasi 1: Precision Rendah (Banyak False Positive)

**Yang terjadi**: Model mendeteksi banyak benda yang bukan kendaraan.

**Cara bikin lebih baik**:
1. Naikkan confidence threshold
   ```bash
   # Ganti di code: conf=0.5 menjadi conf=0.7
   model.predict(source='video.mp4', conf=0.7)
   ```

2. Kumpulkan lebih banyak negative samples (bukan kendaraan)

3. Ulang training dengan penalty lebih tinggi untuk FP

### Situasi 2: Recall Rendah (Melewatkan Kendaraan)

**Yang terjadi**: Model melewatkan banyak kendaraan yang seharusnya terdeteksi.

**Cara bikin lebih baik**:
1. Turunkan confidence threshold
   ```bash
   # Ganti di code: conf=0.5 menjadi conf=0.3
   model.predict(source='video.mp4', conf=0.3)
   ```

2. Kumpulkan data training lebih banyak untuk jenis yang terlewat

3. Tambah augmentation (penggandaan data)
   ```python
   # Di train.py, ubah:
   hsv_v=0.5,    # dari 0.4
   hsv_s=0.5,    # dari 0.4
   mosaic=0.5    # dari 0.2
   ```

### Situasi 3: Satu Jenis Kendaraan Jelek (misal: Bus)

**Yang terjadi**: Mobil dan Truk akurat, tapi Bus jelek.

**Cara bikin lebih baik**:
1. Kumpulkan lebih banyak gambar Bus

2. Tingkatkan bobot loss untuk Bus di training:
   ```python
   model.train(
       data='data.yaml',
       cls_weight=[1.0, 3.0, 1.0]  # 3x weight untuk Bus (class 1)
   )
   ```

3. Ulang training dengan data Bus yang lebih bagus

---

## BERSIAP PAKAI PRODUKSI

### Checklist Sebelum Pakai di Dunia Nyata

```
✅ METRIC CHECKLIST:
  ☐ Precision > 0.75 (75%)
  ☐ Recall > 0.80 (80%)
  ☐ F1-Score > 0.78 (78%)
  ☐ mAP50 > 0.80 (80%)

✅ PERFORMANCE CHECKLIST:
  ☐ Kecepatan deteksi < 0.1 detik per gambar
  ☐ RAM yang digunakan < 2 GB
  ☐ Bisa jalan 30+ gambar per detik

✅ DATA CHECKLIST:
  ☐ Sudah test di minimal 300 gambar
  ☐ Semua jenis kendaraan ada di test data
  ☐ Sudah test di kondisi berbeda (siang, malam, hujan)

✅ OUTPUT CHECKLIST:
  ☐ Evaluasi laporan di-generate ✓
  ☐ Confusion matrix sudah dilihat ✓
  ☐ Per-class metrics semua baik ✓
  ☐ Percaya diri model siap produksi ✓
```

### Jika Semua Checklist Sudah ✓

```
🎉 SELAMAT! Model sudah teruji lokal untuk:
  ✓ Analisis video traffic (lokal folder backend/yolo)
  ✓ Counting kendaraan (akurat 85-88%)
  ✓ Research/Academic project (dengan data valid)
  ✓ Laporan tugas akhir (bersama hasil testing)
  ✓ Defense presentation (dengan metrics)
  ℹ️ Production deployment (perlu adapting lebih lanjut)
```

---

## TROUBLESHOOTING - JIKA ADA ERROR

### Error 1: "No module named 'ultralytics'"

**Penyebab**: Library tidak terinstall

**Solusi**:
```bash
pip install ultralytics
```

### Error 2: "data.yaml not found"

**Penyebab**: File atau path salah

**Solusi**:
```bash
cd backend/yolo  # Pastikan di folder yang tepat
python evaluate_model.py
```

### Error 3: "CUDA out of memory"

**Penyebab**: GPU kehabisan memori

**Solusi**: Edit di evaluate_model.py atau script:
```python
# Turunkan batch size
model.val(batch=2, imgsz=1280)  # atau imgsz=640
```

### Error 4: "Model very slow, FPS low"

**Penyebab**: GPU tidak dipakai atau model terlalu besar

**Solusi**:
```python
# Pastikan pakai GPU
model = YOLO('yolov8n.pt')  # 'n'=nano (paling cepat)
# atau
model = YOLO('yolov8s.pt')  # 's'=small
```

---

## QUICK REFERENCE - PERINTAH CEPAT

```bash
# Test model dengan validation set
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.val(data='data.yaml', imgsz=1280, batch=4)
"

# Deteksi pada 1 gambar
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict(source='foto.jpg', conf=0.5)
"

# Deteksi pada video
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.predict(source='video.mp4', conf=0.5, save_txt=True)
"

# Test dengan confidence berbeda
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
for conf in [0.3, 0.5, 0.7]:
    print(f'Testing dengan conf={conf}')
    model.predict(source='video.mp4', conf=conf)
"
```

---

## RINGKASAN

**Testing Model YOLO - Singkat:**

1. **Jalankan evaluasi** → `python evaluate_model.py`
2. **Lihat metrik** → Precision, Recall, mAP50
3. **Bandingkan dengan target** → Apakah sudah > 0.80?
4. **Jika baik** → Siap pakai!
5. **Jika jelek** → Edit parameter dan latih ulang

**Waktu testing:**
- Test cepat: 5 menit
- Test lengkap: 30 menit
- Analisis & interpretasi: 15-30 menit

---

**dibuat**: 13 Februari 2026  
**Status**: ✅ Siap Digunakan  
**Untuk**: Tugas Akhir Mahasiswa - Test Akurasi Model YOLO
