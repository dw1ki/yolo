import asyncHandler from "../utils/asyncHandler.js";
import YoloResult from "../models/yoloResult.js";
import Perhitungan from "../models/perhitungan.js";

// POST /api/pkji/hitung
export const hitungPKJI = asyncHandler(async (req, res) => {
  const {
    yolo_result_id,
    ruas,
    tipe_jalan,
    jumlah_lajur,
    lebar_lajur,
    fcle,
    durasi_menit
  } = req.body;

  const yolo = await YoloResult.findById(yolo_result_id);
  if (!yolo) return res.status(404).json({ message: "YOLO Result not found" });

  // 1️⃣ Konversi SMP
  const smp_mobil = yolo.mobil * 1;      // faktor default 1
  const smp_bus   = yolo.bus * 2;        // contoh faktor 2
  const smp_truk  = yolo.truk * 3;       // contoh faktor 3
  const smp_total = smp_mobil + smp_bus + smp_truk;

  // 2️⃣ Volume Q
  const Q = smp_total / (durasi_menit / 60);

  // 3️⃣ Kapasitas C
  const C0 = 2000; // kapasitas per lajur contoh
  const C = jumlah_lajur * C0 * fcle;

  // 4️⃣ Derajat Kejenuhan
  const DJ = Q / C;

  // 5️⃣ LOS
  let LOS = "-";
  if(DJ <= 0.6) LOS = "A";
  else if(DJ <= 0.7) LOS = "B";
  else if(DJ <= 0.8) LOS = "C";
  else if(DJ <= 0.9) LOS = "D";
  else if(DJ <= 1.0) LOS = "E";
  else LOS = "F";

  const perhitungan = await Perhitungan.create({
    yolo_result_id,
    ruas,
    tipe_jalan,
    jumlah_lajur,
    lebar_lajur,
    fcle,
    smp_mobil,
    smp_bus,
    smp_truk,
    smp_total,
    Q,
    C,
    DJ,
    LOS,
    status: "FINAL"
  });

  res.json(perhitungan);
});
