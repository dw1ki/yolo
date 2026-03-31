import mongoose from "mongoose";

const perhitunganSchema = new mongoose.Schema({
  // Old fields (for backward compatibility)
  yolo_result_id: { type: mongoose.Schema.Types.ObjectId, ref: "YoloResult" },
  ruas: String,
  tipe_jalan: String,
  jumlah_lajur: Number,
  lebar_lajur: Number,
  fcle: Number,

  smp_mobil: Number,
  smp_bus: Number,
  smp_truk: Number,
  smp_total: Number,

  Q: Number,
  C: Number,
  DJ: Number,
  LOS: String,

  // NEW fields (from frontend PKJI 2023 form)
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  namaRuas: String,
  tipeAlinemen: String, // datar, bukit, gunung
  tipeJalan: String, // 4/2 D, 2/2, etc
  intervalWaktu: String, // format: "HH:mm:ss-HH:mm:ss"
  durasi: String, // format: "HH:MM:SS"
  lajur: String, // 'Kiri' atau 'Kanan'
  
  // Vehicle counts (per lane)
  mobil: Number,
  bus: Number,
  truk: Number,
  smp: Number,
  volume: Number,
  capacity: Number,
  dj: Number,

  // Summary data
  totalVolume: Number,
  djTerberat: Number,
  levelPelayanan: String, // A-F
  kategori: String, // Lancar, Stabil, etc
  deskripsi: String,
  kesimpulan: String,
  tanggal: Date,

  status: { type: String, enum: ["DRAFT","FINAL"], default: "DRAFT" },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

export default mongoose.model("Perhitungan", perhitunganSchema);
