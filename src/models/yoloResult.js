import mongoose from "mongoose";

const yoloResultSchema = new mongoose.Schema({
  video_id: String,
  mobil: Number,
  bus: Number,
  truk: Number,
  lajur_kiri: Number,
  lajur_kanan: Number,
  durasi_menit: Number,
  created_at: { type: Date, default: Date.now },
});

export default mongoose.model("YoloResult", yoloResultSchema);
