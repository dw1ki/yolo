import express from "express";
import Job from "../models/Job.js";

const router = express.Router();

/**
 * GET /jobs
 * Ambil semua histori job (latest first)
 */
router.get("/", async (req, res) => {
  try {
    const jobs = await Job.find().sort({ createdAt: -1 });
    res.json(jobs);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

/**
 * GET /jobs/:id
 * Ambil detail satu job
 */
router.get("/:id", async (req, res) => {
  try {
    const job = await Job.findById(req.params.id);

    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }

    res.json(job);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

export default router;
