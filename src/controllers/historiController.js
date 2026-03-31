import Perhitungan from "../models/perhitungan.js";

export const getHistori = async (req, res) => {
  try {
    const data = await Perhitungan.find()
      .sort({ createdAt: -1 });

    res.json({ status: "success", data });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const getDetail = async (req, res) => {
  try {
    const data = await Perhitungan.findById(req.params.id);

    if (!data) return res.status(404).json({ message: "Not found" });

    res.json({ status: "success", data });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const deleteHistori = async (req, res) => {
  try {
    const data = await Perhitungan.findByIdAndDelete(req.params.id);

    if (!data) return res.status(404).json({ message: "Data not found" });

    res.json({ status: "success", message: "Data deleted successfully", data });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};