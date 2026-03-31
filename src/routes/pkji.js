import express from "express";
import { hitungPKJI } from "../controllers/pkjiController.js";

const router = express.Router();

router.post("/hitung", hitungPKJI);

export default router;
