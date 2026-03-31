import express from "express";
import { getHistori, getDetail, deleteHistori } from "../controllers/historiController.js";

const router = express.Router();

router.get("/", getHistori);
router.get("/:id", getDetail);
router.delete("/:id", deleteHistori);

export default router;
