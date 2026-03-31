import Job from "../models/Job.js";
import Perhitungan from '../models/perhitungan.js';

function hitungLOS(DJ) {
  if (DJ <= 0.35) return "A";
  if (DJ <= 0.54) return "B";
  if (DJ <= 0.77) return "C";
  if (DJ <= 0.93) return "D";
  if (DJ <= 1.0) return "E";
  return "F";
}

export const hitungPerhitungan = async (req, res) => {
  try {
    const {
      jobId,
      n_lajur,
      C0,
      FCLE,
      lebar_lajur,
      tipe_jalan
    } = req.body;

    // ambil data YOLO
    const job = await Job.findById(jobId);
    if (!job) return res.status(404).json({ message: "Job not found" });

    const volume = job.result.result;

    const totalVolume = volume.total;

    // === RUMUS PKJI ===
    const C = n_lajur * C0 * FCLE;
    const DJ = totalVolume / C;
    const LOS = hitungLOS(DJ);

    const data = await Perhitungan.create({
      jobId,
      n_lajur,
      C0,
      FCLE,
      lebar_lajur,
      tipe_jalan,
      volume: {
        mobil: volume.kiri.mobil + volume.kanan.mobil,
        bus: volume.kiri.bus + volume.kanan.bus,
        truk: volume.kiri.truk + volume.kanan.truk,
        total: totalVolume
      },
      C,
      DJ,
      LOS
    });

    res.json({ status: "success", data });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

// NEW: Save Perhitungan dari frontend dengan per-lane data
export const savePerhitungan = async (req, res) => {
  try {
    const {
      namaRuas,
      tipeAlinemen,
      tipeJalan,
      intervalWaktu,
      durasi,
      lajur,
      mobil,
      bus,
      truk,
      smp,
      volume,
      capacity,
      dj,
      totalVolume,
      djTerberat,
      levelPelayanan,
      kategori,
      deskripsi,
      kesimpulan,
      tanggal
    } = req.body;

    const userId = req.user?.id; // dari middleware auth

    const data = await Perhitungan.create({
      userId,
      namaRuas,
      tipeAlinemen,
      tipeJalan,
      intervalWaktu,
      durasi,
      lajur,
      mobil,
      bus,
      truk,
      smp,
      volume,
      capacity,
      dj,
      totalVolume,
      djTerberat,
      levelPelayanan,
      kategori,
      deskripsi,
      kesimpulan,
      tanggal: new Date(tanggal)
    });

    res.json({ success: true, message: "Perhitungan berhasil disimpan", data });

  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// GET: Fetch user's perhitungan history
export const getPerhitunganHistory = async (req, res) => {
  try {
    const userId = req.user?.id;

    const data = await Perhitungan.find({ userId })
      .sort({ createdAt: -1 })
      .select('-__v');

    res.json({
      success: true,
      data: data || []
    });

  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// DELETE: Remove perhitungan record
export const deletePerhitungan = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;

    const record = await Perhitungan.findById(id);
    if (!record) {
      return res.status(404).json({ success: false, message: "Record not found" });
    }

    // Check if user owns this record
    if (record.userId.toString() !== userId) {
      return res.status(403).json({ success: false, message: "Unauthorized" });
    }

    await Perhitungan.findByIdAndDelete(id);

    res.json({
      success: true,
      message: "Record deleted successfully"
    });

  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
