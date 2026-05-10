import express from "express";
import axios from "axios";
import FormData from "form-data";
import jwt from "jsonwebtoken";
import upload from "../middlewares/uploadVideo.js";
import {
  uploadVideo,
  processVideo,
  saveParameters,
  calculateResults,
  getDetectionResults,
  getDetectionHistory,
  updateDetectionStatus,
  deleteDetection,
  getJobStatus,
  saveYOLOResults,
} from "../controllers/detectController.js";
import { protect, authorize } from "../middlewares/auth.js";

// Simple middleware for file upload to local storage
function getUploadMiddleware() {
  return upload.single("video");
}

const router = express.Router();

// ⭐ NEW: Health check function untuk verify YOLO API availability
async function checkYOLOHealth() {
  try {
    const pythonApiUrl = process.env.PYTHON_API || "https://hurtling-unforecasted-horace.ngrok-free.dev";
    const apiUrl = pythonApiUrl.endsWith("/") ? pythonApiUrl.slice(0, -1) : pythonApiUrl;
    
    const healthRes = await axios.get(`${apiUrl}/health`, { 
      timeout: 5000 
    });
    
    const isHealthy = healthRes.status === 200 && healthRes.data?.status === "healthy";
    return {
      healthy: isHealthy,
      status: healthRes.data?.status,
      device: healthRes.data?.device,
      message: isHealthy ? "YOLO API is ready" : "YOLO API returned non-healthy status"
    };
  } catch (err) {
    return {
      healthy: false,
      status: "unreachable",
      error: err.message,
      message: `YOLO API health check failed: ${err.message}`
    };
  }
}

// ⭐ NEW: Middleware untuk check YOLO health sebelum process
async function checkYOLOBeforeProcess(req, res, next) {
  try {
    console.log(`🏥 [YOLO Health] Checking API health before processing...`);
    const health = await checkYOLOHealth();
    
    if (!health.healthy) {
      console.warn(`⚠️  [YOLO Health] API is not healthy:`, health);
      return res.status(503).json({
        success: false,
        error: "YOLO API unavailable",
        details: health.message,
        status: "service_unavailable"
      });
    }
    
    console.log(`✅ [YOLO Health] API is healthy, proceeding...`);
    next();
  } catch (err) {
    console.error(`❌ [YOLO Health] Unexpected error during health check:`, err.message);
    return res.status(500).json({
      success: false,
      error: "Failed to check YOLO API health",
      details: err.message
    });
  }
}

// ===== MOCK ENDPOINT - For quick testing without YOLO =====
router.post("/yolo/mock", protect, async (req, res) => {
  try {
    console.log("🎭 [Backend] Using MOCK YOLO endpoint for testing...");
    
    // Return mock YOLO response
    return res.json({
      success: true,
      job_id: `mock_job_${Date.now()}`,
      video_url: "https://via-mock/video.mp4",
      cloudinary_url: "https://via-mock/cloudinary.mp4",
      message: "This is mock data - real YOLO processing not available"
    });
  } catch (err) {
    console.error("❌ [Backend] Mock endpoint error:", err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ===== YOLO PROXY ROUTES (BEFORE protect middleware) =====

// POST /api/detect/yolo/process - Upload video ke YOLO API (dari backend)
router.post("/yolo/process", protect, getUploadMiddleware(), async (req, res) => {
  try {
    console.log("🔄 [YOLO] Process endpoint called");
    console.log("🔄 [YOLO] User ID:", req.userId);
    
    if (!req.file) {
      console.log("❌ [YOLO] No file uploaded");
      return res.status(400).json({ error: "No video file uploaded" });
    }

    const fileName = req.file.filename || req.file.originalname;
    const fileSize = req.file.size;
    const cloudinaryUrl = req.file.path;

    console.log("✅ [YOLO] File received from Cloudinary");
    console.log("   Name:", fileName);
    console.log("   Size:", fileSize, "bytes");
    console.log("   URL:", cloudinaryUrl);

    // Download dari Cloudinary
    console.log("📥 [YOLO] Downloading dari Cloudinary...");
    let videoBuffer;
    try {
      const videoRes = await axios.get(cloudinaryUrl, {
        responseType: "arraybuffer",
        timeout: 60000,
      });
      videoBuffer = Buffer.from(videoRes.data);
      console.log(`✅ [YOLO] Downloaded: ${videoBuffer.length} bytes`);
    } catch (downloadErr) {
      console.error("❌ [YOLO] Cloudinary download failed:", downloadErr.message);
      return res.status(500).json({ 
        error: "Failed to download from Cloudinary",
        details: downloadErr.message,
        url: cloudinaryUrl
      });
    }

    // Upload ke YOLO dengan extended timeout
    console.log("📤 [YOLO] Preparing upload to Railway YOLO...");
    const form = new FormData();
    form.append("file", videoBuffer, fileName);

    const pythonApiUrl = process.env.PYTHON_API || "https://hurtling-unforecasted-horace.ngrok-free.dev";
    console.log(`📤 [YOLO] Target API: ${pythonApiUrl}/detect`);

    let yoloRes;
    try {
      yoloRes = await axios.post(
        `${pythonApiUrl}/detect`,
        form,
        {
          headers: form.getHeaders(),
          timeout: 600000, // 10 menit - YOLO bisa butuh waktu lama untuk upload besar
          maxRedirects: 5,
        }
      );
      console.log("✅ [YOLO] YOLO API response received");
      console.log("✅ [YOLO] Response data:", JSON.stringify(yoloRes.data).substring(0, 200));
    } catch (yoloErr) {
      console.error("❌ [YOLO] YOLO API error:");
      console.error("   Message:", yoloErr.message);
      console.error("   Status:", yoloErr.response?.status);
      console.error("   Response:", yoloErr.response?.data);
      console.error("   Code:", yoloErr.code);
      return res.status(500).json({
        error: "YOLO API failed",
        details: yoloErr.message,
        status: yoloErr.response?.status,
        yoloResponse: yoloErr.response?.data,
      });
    }

    if (!yoloRes.data || !yoloRes.data.job_id) {
      console.error("❌ [YOLO] Invalid YOLO response - no job_id:", yoloRes.data);
      return res.status(500).json({ 
        error: "Invalid YOLO response",
        details: "No job_id returned",
        received: yoloRes.data
      });
    }

    console.log("✅ [YOLO] Success! Job ID:", yoloRes.data.job_id);

    return res.json({
      success: true,
      job_id: yoloRes.data.job_id,
      video_url: yoloRes.data.video_url || cloudinaryUrl,
      cloudinary_url: cloudinaryUrl,
    });
  } catch (err) {
    console.error("❌ [Backend] Unexpected error:");
    console.error("   Message:", err.message);
    console.error("   Stack:", err.stack.split('\n')[0]);
    console.error("   Stack:", err.stack);
    
    return res.status(500).json({
      error: "Failed to process video with YOLO",
      details: err.message,
      yoloStatus: err.response?.status,
      yoloData: err.response?.data,
    });
  }
});

// GET /api/detect/yolo/result/:jobId - Poll YOLO result
router.get("/yolo/result/:jobId", verifyToken, async (req, res) => {
  try {
    const jobId = req.params.jobId;
    console.log(`📊 [YOLO] Polling result for job: ${jobId}`);
    console.log(`🔑 [YOLO] User ID: ${req.userId}`);
    
    const pythonApiUrl = process.env.PYTHON_API || "https://hurtling-unforecasted-horace.ngrok-free.dev";
    const url = `${pythonApiUrl}/result/${jobId}`;
    
    console.log(`📊 [YOLO] Target API: ${url}`);
    
    let yoloRes;
    try {
      yoloRes = await axios.get(url, { 
        timeout: 10000,
        validateStatus: (status) => status < 500 // Accept 4xx too
      });
      console.log(`📊 [YOLO] API Response status: ${yoloRes.status}`);
      console.log(`📊 [YOLO] API Response data:`, JSON.stringify(yoloRes.data).substring(0, 300));
    } catch (pollErr) {
      console.error(`❌ [YOLO] Poll error:`, pollErr.message);
      console.error(`❌ [YOLO] Status:`, pollErr.response?.status);
      console.error(`❌ [YOLO] Data:`, pollErr.response?.data);
      throw pollErr;
    }

    return res.json(yoloRes.data);
  } catch (err) {
    console.error(`❌ [Backend] YOLO result poll error for job ${req.params.jobId}:`, err.message);
    
    if (err.response?.status === 404) {
      console.log(`ℹ️ [YOLO] Job not found or still processing`);
      return res.status(404).json({ error: "Job not found or still processing" });
    }
    return res.status(500).json({
      error: "Failed to get YOLO result",
      details: err.message,
      jobId: req.params.jobId
    });
  }
});

// GET /api/detect/yolo/frame/:jobId - Get current processing frame/thumbnail dari YOLO
router.get("/yolo/frame/:jobId", verifyToken, async (req, res) => {
  try {
    const jobId = req.params.jobId;
    console.log(`📸 [YOLO] Requesting frame for job: ${jobId}`);
    
    const pythonApiUrl = process.env.PYTHON_API || "https://hurtling-unforecasted-horace.ngrok-free.dev";
    const url = `${pythonApiUrl}/frame/${jobId}`;
    
    console.log(`📸 [YOLO] Target API: ${url}`);
    
    let yoloRes;
    try {
      yoloRes = await axios.get(url, { 
        timeout: 10000,
        responseType: "stream",
        validateStatus: (status) => status < 500
      });
      console.log(`📸 [YOLO] Frame response status: ${yoloRes.status}`);
      
      // Proxy frame response
      res.setHeader("Content-Type", yoloRes.headers["content-type"] || "image/jpeg");
      yoloRes.data.pipe(res);
    } catch (frameErr) {
      console.error(`❌ [YOLO] Frame fetch error:`, frameErr.message);
      throw frameErr;
    }
  } catch (err) {
    console.error(`❌ [Backend] YOLO frame fetch error for job ${req.params.jobId}:`, err.message);
    return res.status(500).json({
      error: "Failed to get YOLO frame",
      details: err.message,
      jobId: req.params.jobId
    });
  }
});

// ===== Original routes =====

// ⭐ NEW: Health check endpoint
router.get("/health/yolo", protect, async (req, res) => {
  try {
    const health = await checkYOLOHealth();
    if (health.healthy) {
      return res.json({ success: true, ...health });
    } else {
      return res.status(503).json({ success: false, ...health });
    }
  } catch (err) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

// Save YOLO results (dari Railway) - need authentication
router.post("/", protect, saveYOLOResults);

// Upload video to local storage
router.post("/upload", protect, getUploadMiddleware(), uploadVideo);

// Process video (send ke YOLO API)
// ⭐ ADDED: Health check middleware before processing
router.post("/process", protect, checkYOLOBeforeProcess, processVideo);

// Save road parameters
router.put("/:id/parameters", saveParameters);

// Calculate results
router.post("/:id/calculate", calculateResults);

// Get detection results
router.get("/results/:id", getDetectionResults);
// ===== HELPER: Create test detection with exact per-lane structure =====
router.post("/test/create-sample", verifyToken, async (req, res) => {
  try {
    console.log("🧪 [Test] Creating sample detection with exact per-lane data...");
    
    const Detection = (await import("../models/Detection.js")).default;
    
    const testData = {
      userId: req.userId,
      videoUrl: "https://res.cloudinary.com/test/video.mp4",
      cloudinaryPublicId: "test_sample_video",
      cloudinarySecureUrl: "https://res.cloudinary.com/test/video.mp4",
      fileName: "Test Sample - Per Lane",
      videoDuration: 300,
      status: "draft",
      yoloResults: {
        totalVehicles: 445,
        avgConfidence: 0.87,
        vehicleTypes: {
          mobilPenumpang: 356,
          bus: 67,
          truckRingan: 22,
          truckBerat: 0,
        },
        totalFrames: 4499,
        leftLaneCount: 245,
        rightLaneCount: 200,
        leftLane: {
          mobil: 196,
          bus: 37,
          truk: 12,
        },
        rightLane: {
          mobil: 160,
          bus: 30,
          truk: 10,
        },
      },
    };
    
    const detection = new Detection(testData);
    await detection.save();
    
    console.log("✅ [Test] Sample detection created with ID:", detection._id);
    
    res.json({
      success: true,
      message: "Test detection created successfully",
      data: {
        _id: detection._id,
        fileName: detection.fileName,
        yoloResults: detection.yoloResults,
      },
    });
  } catch (err) {
    console.error("❌ [Test] Error:", err.message);
    res.status(500).json({
      success: false,
      message: "Error creating test detection",
      error: err.message,
    });
  }
});

// Get detection history
router.get("/history", getDetectionHistory);

// ===== HELPER: Seed sample per-lane data for testing =====
router.post("/seed/sample-detection", verifyToken, async (req, res) => {
  try {
    console.log("🌱 [Seed] Creating sample detection with per-lane data...");
    
    const Detection = (await import("../models/Detection.js")).default;
    
    const sampleDetection = new Detection({
      userId: req.userId,
      videoUrl: "https://res.cloudinary.com/sample/video.mp4",
      cloudinaryPublicId: "sample_video",
      cloudinarySecureUrl: "https://res.cloudinary.com/sample/video.mp4",
      fileName: "Sample YOLO Detection",
      videoDuration: 300,
      status: "draft",
      yoloResults: {
        totalVehicles: 82,
        avgConfidence: 0.87,
        vehicleTypes: {
          mobilPenumpang: 66,
          bus: 14,
          truckRingan: 2,
          truckBerat: 0,
        },
        totalFrames: 4499,
        leftLaneCount: 45,
        rightLaneCount: 37,
        leftLane: {
          mobil: 36,
          bus: 8,
          truk: 1,
        },
        rightLane: {
          mobil: 30,
          bus: 6,
          truk: 1,
        },
      },
    });
    
    await sampleDetection.save();
    
    console.log("✅ [Seed] Sample detection created:", sampleDetection._id);
    
    res.json({
      success: true,
      message: "Sample detection created for testing",
      data: sampleDetection,
    });
  } catch (err) {
    console.error("❌ [Seed] Error:", err.message);
    res.status(500).json({
      success: false,
      message: "Error creating sample detection",
      error: err.message,
    });
  }
});

// Update detection status
router.put("/:id/status", updateDetectionStatus);

// Delete detection
router.delete("/:id", protect, deleteDetection);

// Backward compatibility - Get job status
router.get("/status/:id", getJobStatus);

// ===== Apply protect middleware ONLY to remaining routes =====
router.use(protect);

// (All remaining routes already defined above)

export default router;

