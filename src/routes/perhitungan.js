import express from "express";
import { hitungPerhitungan, savePerhitungan, getPerhitunganHistory, deletePerhitungan } from "../controllers/perhitunganController.js";
import { protect } from "../middlewares/auth.js";

const router = express.Router();

router.post("/", hitungPerhitungan);
router.post("/save", protect, savePerhitungan);
router.get("/history", protect, getPerhitunganHistory);
router.delete("/:id", protect, deletePerhitungan);

export default router;
