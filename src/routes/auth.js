import express from "express";
import {
  register,
  login,
  refreshAccessToken,
  getMe,
  logout,
} from "../controllers/authController.js";
import { protect, authorize } from "../middlewares/auth.js";

const router = express.Router();

// Public routes
router.post("/register", register);
router.post("/login", login);
router.post("/refresh", refreshAccessToken);

// Protected routes
router.get("/me", protect, getMe);
router.post("/logout", protect, logout);

export default router;
