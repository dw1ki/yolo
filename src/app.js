import dotenv from "dotenv";
import express from "express";
import cookieParser from "cookie-parser";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Load env FIRST
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// ==================== CORS FIRST - BEFORE EVERYTHING ====================
const allowedOrigins = [
  'https://pktj.vercel.app',
  'https://pktj.netlify.app',
  'http://localhost:3000',
  'http://localhost:5173',
];

app.use((req, res, next) => {
  const origin = req.headers.origin;
  
  // Set CORS headers - be more permissive to debug
  if (origin && allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  } else if (!origin) {
    // Local or no origin
    res.setHeader('Access-Control-Allow-Origin', '*');
  } else {
    // Allow anyway for debugging - remove in production
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, PUT, PATCH, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  
  // Handle ALL OPTIONS requests immediately - don't process further
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  next();
});

// ==================== Middleware ====================
// Increase timeout untuk long-running jobs (1 jam+ processing)
app.use(express.json({ 
  limit: "50mb",
  timeout: "600000ms" // 10 menit timeout
}));
app.use(express.urlencoded({ 
  limit: "50mb", 
  extended: true,
  timeout: "600000ms" // 10 menit timeout
}));
app.use(cookieParser());

// Set server timeout ke 15 menit untuk long-running requests
app.set('json spaces', 2);

// ================== Routes - Import at startup ==================
// Import routes ONCE at startup to maintain singleton instances (like jobQueue)
import authRoutes from "./routes/auth.js";
import userRoutes from "./routes/user.js";
import detectRoutes from "./routes/detectNew.js";
import dashboardRoutes from "./routes/dashboard.js";
import perhitunganRoutes from "./routes/perhitungan.js";
import historiRoutes from "./routes/histori.js";
import jobsRoutes from "./routes/jobs.js";

app.use("/api/auth", authRoutes);
app.use("/api/users", userRoutes);
app.use("/api/detect", detectRoutes);
app.use("/api/dashboard", dashboardRoutes);
app.use("/api/perhitungan", perhitunganRoutes);
app.use("/api/histori", historiRoutes);
app.use("/jobs", jobsRoutes);

// ================== TEST ENDPOINTS - Always work ==================
app.get("/", (req, res) => {
  res.json({ status: "Backend online" });
});

app.get("/health", (req, res) => {
  res.json({ status: "healthy" });
});

// ==================== Error Handler ====================
app.use((err, req, res, next) => {
  const origin = req.headers.origin;
  if (!origin || allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin || '*');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, PUT, PATCH, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  
  console.error('[Error]', err.message);
  
  res.status(err.statusCode || 500).json({
    success: false,
    message: err.message || "Server error",
  });
});

// ==================== 404 Fallback ====================
app.use((req, res) => {
  const origin = req.headers.origin;
  if (!origin || allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin || '*');
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, PUT, PATCH, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  
  res.status(404).json({
    success: false,
    message: "Endpoint not found",
  });
});

export default app;

