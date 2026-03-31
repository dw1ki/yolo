import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";

import jobRoutes from "./src/routes/jobs.js";
import detectRoutes from "./src/routes/detect.js";

import historiRoutes from "./src/routes/histori.js";
import userRoutes from "./src/routes/user.js";
import authRoutes from "./src/routes/auth.js";

dotenv.config();

const app = express();

/* =====================
   MIDDLEWARE
===================== */
app.use(cors({
  origin: [
    "https://8jhzzjcv-3000.asse.devtunnels.ms", // frontend devtunnel
    "http://localhost:3000" // local dev
  ],
  credentials: true
}));
app.use(express.json());

console.log("🔥 Backend starting...");

/* =====================
   ROUTES
===================== */
console.log("🔥 Registering routes...");

// Register auth routes for /api/auth
app.use("/api/auth", authRoutes);

app.use("/jobs", jobRoutes);

// Register histori routes for /api/histori
app.use("/api/histori", historiRoutes);

// Register user routes for /api/users
app.use("/api/users", userRoutes);

/**
 * ⚠️ PENTING
 * Endpoint detect = POST /api/detect
 * JANGAN pakai /api doang
 */
app.use("/api/detect", detectRoutes);

/* =====================
   ROOT TEST
===================== */
app.get("/", (req, res) => {
  res.json({ status: "Backend running 🚀" });
});

/* =====================
   SERVER + DATABASE
===================== */
const PORT = process.env.PORT || 5000;

// ⛔ JANGAN listen sebelum MongoDB connect
async function startServer() {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("✅ MongoDB connected");

    app.listen(PORT, () => {
      console.log(`🚀 Backend running on http://localhost:${PORT}`);
      console.log(`📌 Detect endpoint: POST /api/detect`);
    });
  } catch (err) {
    console.error("❌ MongoDB connection failed:", err.message);
    process.exit(1);
  }
}

startServer();
