import Detection from "../models/Detection.js";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { saveToLocalStorage } from "../middlewares/uploadVideo.js";
import {
  performCalculation,
  validateRoadParameters,
  validateYOLOResults,
} from "../utils/calculation.js";

// ⭐ Define __dirname untuk ES6 modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ================== UPLOAD VIDEO TO LOCAL STORAGE ==================
export const uploadVideo = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "File video wajib diunggah",
      });
    }

    console.log(`📹 [UPLOAD] Received file: ${req.file.originalname} (${req.file.size} bytes)`);

    // Validate file type
    const allowedMimes = ["video/mp4", "video/mpeg", "video/quicktime"];
    if (!allowedMimes.includes(req.file.mimetype)) {
      return res.status(400).json({
        success: false,
        message: "Format video tidak didukung. Gunakan MP4, MPEG, atau MOV",
      });
    }

    // Validate file size (no limit now - local storage)
    if (req.file.size > 5 * 1024 * 1024 * 1024) {
      return res.status(400).json({
        success: false,
        message: "Ukuran file terlalu besar. Maksimal 5GB",
      });
    }

    // ⭐ NEW: Save to local storage instead of Cloudinary
    console.log(`💾 [UPLOAD] Saving to local storage...`);
    const localFileInfo = await saveToLocalStorage(req.file.path, req.file.originalname);
    console.log(`✅ [UPLOAD] File saved successfully to: ${localFileInfo.filePath}`);

    // Verify file exists before saving to DB
    if (!fs.existsSync(localFileInfo.filePath)) {
      throw new Error(`File verification failed - file not found immediately after save at: ${localFileInfo.filePath}`);
    }
    console.log(`✓ [UPLOAD] File existence verified: ${localFileInfo.fileSize} bytes`);

    // Create detection record with local file path
    const detection = new Detection({
      userId: req.user._id,
      videoUrl: localFileInfo.filePath, // ⭐ LOCAL PATH
      fileName: req.file.originalname,
      fileSize: localFileInfo.fileSize,
      status: "draft",
      createdBy: req.user.name,
      cloudinaryPublicId: null,
      cloudinarySecureUrl: null,
      storageType: "local", // ⭐ Track that this is local storage
    });

    await detection.save();
    console.log(`✅ [UPLOAD] Detection record created: ${detection._id}`);

    res.status(201).json({
      success: true,
      message: "Video berhasil diunggah ke penyimpanan lokal",
      data: {
        id: detection._id,
        videoUrl: detection.videoUrl,
        fileName: detection.fileName,
        fileSize: detection.fileSize,
        storageType: "local",
        status: detection.status,
        createdAt: detection.createdAt,
      },
    });
  } catch (error) {
    console.error(`❌ [UPLOAD] Error:`, error);
    res.status(500).json({
      success: false,
      message: "Gagal mengupload video",
      error: error.message,
    });
  }
};

// ================== PROCESS VIDEO WITH YOLO API ==================
export const processVideo = async (req, res) => {
  try {
    const { detectionId, recordingInterval, videoDuration } = req.body;

    if (!detectionId) {
      return res.status(400).json({
        success: false,
        message: "Detection ID wajib diisi",
      });
    }

    // Find detection record
    const detection = await Detection.findById(detectionId);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Check ownership
    if (detection.userId.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: "Anda tidak memiliki akses ke detection ini",
      });
    }

    if (detection.status !== "draft") {
      return res.status(400).json({
        success: false,
        message: `Detection sudah dalam status ${detection.status}`,
      });
    }

    // Update detection with metadata
    detection.status = "processing";
    detection.recordingInterval = recordingInterval || "";
    detection.videoDuration = videoDuration || 0;
    await detection.save();

    // Send to YOLO API asynchronously
    processWithYOLO(detection._id, detection.videoUrl).catch((error) => {
      console.error("YOLO API processing error:", error);
    });

    res.json({
      success: true,
      message: "Video sedang diproses oleh YOLO API. Ini akan memakan waktu beberapa menit...",
      data: {
        id: detection._id,
        status: "processing",
      },
    });
  } catch (error) {
    console.error("Process video error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal memproses video",
      error: error.message,
    });
  }
};

// ================== YOLO API PROCESSING (ASYNC) ==================
const processWithYOLO = async (detectionId, videoUrl) => {
  try {
    // ⭐ PREFER LOCALHOST for local development (direct, no tunnel delays)
    // Fallback to PYTHON_API (Vercel) or YOLO_API_URL (Railway) for production
    let pythonApiUrl = process.env.PYTHON_API || process.env.YOLO_API_URL;
    let apiSource = "env (PYTHON_API / YOLO_API_URL)";

    // Try localhost first if running locally (avoids cloudflared tunnel timeouts)
    if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
      try {
        const localhostResponse = await axios.head("http://localhost:8000/status", { timeout: 2000 });
        if (localhostResponse.status === 200 || localhostResponse.status === 404) {
          pythonApiUrl = "http://localhost:8000";
          apiSource = "localhost (local YOLO server)";
          console.log(`✅ [PROCESS] Detected local YOLO server, using direct connection`);
        }
      } catch (err) {
        // localhost not available, use configured URL
        console.log(`ℹ️  [PROCESS] Local YOLO server not reachable (${err.code}), falling back to configured API`);
      }
    }

    if (!pythonApiUrl) {
      throw new Error("YOLO API tidak dikonfigurasi. Set PYTHON_API atau YOLO_API_URL di environment variables");
    }

    console.log(`\n🔄 [PROCESS] ========== PROCESSING STARTED ==========`);
    console.log(`📋 Detection ID: ${detectionId}`);
    console.log(`📁 Local video path: ${videoUrl}`);
    console.log(`🌐 Using YOLO API: ${pythonApiUrl} (${apiSource})`);

    // ⭐ VERIFY FILE EXISTS - check if it's local path or needs conversion
    let actualVideoPath = videoUrl;
    
    // If path starts with /app (Railway container path), convert to local path
    if (actualVideoPath.startsWith("/app")) {
      // Container path detected - convert to local Windows/Unix path
      console.log(`⚠️  [PROCESS] Container path detected: ${actualVideoPath}`);
      
      // Extract relative path after /app/
      const relativePath = actualVideoPath.replace("/app/", "");
      const basePath = process.env.NODE_ENV === "production" 
        ? path.join(__dirname, "../../") 
        : "D:\\new\\pktj\\backend";
      
      actualVideoPath = path.join(basePath, relativePath);
      console.log(`📝 [PROCESS] Converted to local path: ${actualVideoPath}`);
    }
    
    if (!fs.existsSync(actualVideoPath)) {
      throw new Error(`❌ Video file not found at: ${actualVideoPath} (original: ${videoUrl})`);
    }
    const fileStats = fs.statSync(actualVideoPath);
    console.log(`✓ [PROCESS] File verified: ${(fileStats.size / 1024 / 1024).toFixed(2)} MB`);

    // ⭐ SEND FILE TO YOLO (multipart upload)
    // Since YOLO is on local machine via ngrok, it cannot access Railway container paths
    // Must send file content, not just path
    console.log(`📤 [PROCESS] Uploading file to YOLO API via multipart...`);
    console.log(`📤 [PROCESS] File size: ${(fileStats.size / 1024 / 1024).toFixed(2)} MB`);
    
    // Create FormData with file stream
    const fileStream = fs.createReadStream(actualVideoPath);
    const fileName = actualVideoPath.split("/").pop() || actualVideoPath.split("\\").pop();
    
    const form = new FormData();
    form.append("file", fileStream, fileName);
    form.append("file_detection_id", detectionId.toString());
    
    const detectResponse = await axios.post(
      `${pythonApiUrl}/detect`,
      form,
      {
        headers: form.getHeaders(),
        timeout: 300000, // 5 minutes for upload
      }
    );

    console.log(`✅ [PROCESS] YOLO accepted job:`, detectResponse.data);
    const jobId = detectResponse.data.job_id || detectionId.toString();

    // ⭐ STEP 2: Poll the /result endpoint until job completes
    console.log(`\n🔄 [PROCESS] ========== POLLING STARTED ==========`);
    console.log(`📊 Polling YOLO API for job ${jobId}...`);
    let jobResult = null;
    let attempts = 0;
    let consecutiveErrors = 0;
    const maxAttempts = 3600; // 1 hour with 1-second polls = 3600 seconds
    const maxConsecutiveErrors = 20; // Allow up to 20 consecutive transient errors before giving up

    while (attempts < maxAttempts) {
      try {
        const resultResponse = await axios.get(
          `${pythonApiUrl}/result/${jobId}`,
          {
            timeout: 600000, // 10 minutes per poll (for long videos - 30min video takes 55min to process!)
          }
        );

        jobResult = resultResponse.data;
        const status = jobResult.status;
        consecutiveErrors = 0; // Reset error counter on success

        if (attempts % 10 === 0) {
          // Log every 10 attempts
          console.log(
            `[Poll ${attempts}] Status: ${status}, Progress: ${jobResult.progress}%, Vehicles: ${jobResult.vehicle_count}`
          );
        }

        if (status === "completed" || status === "done") {
          console.log(`✅ [PROCESS] ========== YOLO COMPLETED ==========`);
          console.log(`✓ Total time: ${(attempts * 1000 / 60000).toFixed(1)} minutes`);
          console.log(`✓ Vehicles detected: ${jobResult.vehicle_count}`);
          break;
        } else if (status === "failed" || status === "error") {
          throw new Error(`YOLO job failed: ${jobResult.message || "Unknown error"}`);
        }

        // Still processing, wait before next poll
        await new Promise((r) => setTimeout(r, 1000));
        attempts++;
      } catch (pollErr) {
        consecutiveErrors++;
        const errorCode = pollErr.code || pollErr.response?.status || pollErr.message;
        const respStatus = pollErr.response?.status;
        const isTransientError = 
          pollErr.code === 'ECONNRESET' ||
          pollErr.code === 'ECONNREFUSED' ||
          pollErr.code === 'ETIMEDOUT' ||
          pollErr.code === 'ENOTFOUND' ||
          // Treat gateway/timeouts and server errors as transient (include Cloudflare 524)
          respStatus === 504 ||
          respStatus === 503 ||
          respStatus === 502 ||
          respStatus === 524 ||
          (typeof respStatus === 'number' && respStatus >= 500 && respStatus < 600) ||
          // 404 may be transient if the remote service is still initializing
          respStatus === 404 ||
          pollErr.message?.includes('SSL') ||
          pollErr.message?.includes('ssl');

        // Log every 5 attempts or on first error
        if (consecutiveErrors === 1 || consecutiveErrors % 5 === 0) {
          console.warn(
            `⚠️  Poll attempt ${attempts} failed (status=${respStatus} / code=${pollErr.code}): ${consecutiveErrors}/${maxConsecutiveErrors} consecutive errors - ${pollErr.message}`,
            { responseData: pollErr.response?.data }
          );
        }

        // If transient error, retry with backoff
        if (isTransientError && consecutiveErrors < maxConsecutiveErrors) {
          const backoffDelay = Math.min(1000 + (consecutiveErrors * 500), 10000); // 1s to 10s backoff
          if (consecutiveErrors === 1) {
            console.log(`ℹ️  Transient error detected, will retry with exponential backoff...`);
          }
          await new Promise((r) => setTimeout(r, backoffDelay));
          attempts++;
        } else if (consecutiveErrors >= maxConsecutiveErrors) {
          // Too many consecutive errors, give up
          const lastResp = pollErr.response ? `status=${pollErr.response.status}` : '';
          throw new Error(
            `YOLO polling failed after ${consecutiveErrors} consecutive transient errors. ` +
            `Last error: ${errorCode} ${lastResp} - ${pollErr.message}. ` +
            `Check ngrok/tunnel status and network connectivity.`
          );
        } else {
          // Non-transient error, throw immediately
          throw pollErr;
        }
      }
    }

    if (!jobResult || jobResult.status !== "completed") {
      throw new Error(
        `YOLO job did not complete within ${maxAttempts} seconds or timed out`
      );
    }

    // ⭐ DEBUG: Log jobResult to trace outputVideoUrl
    console.log(`[DEBUG] jobResult from Python API:`, {
      status: jobResult.status,
      vehicle_count: jobResult.vehicle_count,
      outputVideoUrl: jobResult.outputVideoUrl,
      backendUrl: jobResult.backendUrl,
      cloudinaryUrl: jobResult.cloudinaryUrl
    });

    // ⭐ STEP 3: Update detection with YOLO results
    const detection = await Detection.findById(detectionId);

    detection.status = "completed";
    detection.yoloResults = {
      totalVehicles: jobResult.vehicle_count || 0,
      vehicleTypes: {
        mobil: 0, // Will be populated from lane data if available
        bus: jobResult.lane?.kiri?.bus + jobResult.lane?.kanan?.bus || 0,
        truk: 0,
      },
      volumeSMP: 0,
      avgConfidence: 0.85,
      totalFrames: jobResult.frames_processed || 0,
      rawData: jobResult,
      // NEW: Per-lane data
      leftLaneCount: jobResult.lane?.kiri?.total || 0,
      rightLaneCount: jobResult.lane?.kanan?.total || 0,
      leftLane: {
        mobil: jobResult.lane?.kiri?.mobil || 0,
        bus: jobResult.lane?.kiri?.bus || 0,
        truk: jobResult.lane?.kiri?.truk || 0,
      },
      rightLane: {
        mobil: jobResult.lane?.kanan?.mobil || 0,
        bus: jobResult.lane?.kanan?.bus || 0,
        truk: jobResult.lane?.kanan?.truk || 0,
      },
    };

    // Calculate total SMP with proper classification
    const leftSMP =
      (jobResult.lane?.kiri?.mobil || 0) * 1.0 +
      (jobResult.lane?.kiri?.bus || 0) * 1.3 +
      (jobResult.lane?.kiri?.truk || 0) * 1.5;

    const rightSMP =
      (jobResult.lane?.kanan?.mobil || 0) * 1.0 +
      (jobResult.lane?.kanan?.bus || 0) * 1.3 +
      (jobResult.lane?.kanan?.truk || 0) * 1.5;

    detection.yoloResults.volumeSMP = leftSMP + rightSMP;
    detection.yoloResults.vehicleTypes.mobil =
      (jobResult.lane?.kiri?.mobil || 0) + (jobResult.lane?.kanan?.mobil || 0);
    detection.yoloResults.vehicleTypes.truk =
      (jobResult.lane?.kiri?.truk || 0) + (jobResult.lane?.kanan?.truk || 0);
    
    // ⭐ ADD OUTPUT VIDEO URL FOR FRONTEND
    detection.yoloResults.outputVideoUrl = jobResult.backendUrl || jobResult.outputVideoUrl || `/download/${detectionId}`;
    detection.videoUrl = detection.yoloResults.outputVideoUrl; // Also set on detection for backward compatibility

    await detection.save();

    console.log(
      `Detection ${detectionId} processed successfully. Total: ${detection.yoloResults.totalVehicles} vehicles, SMP: ${detection.yoloResults.volumeSMP}`
    );
  } catch (error) {
    console.error(`YOLO API error for detection ${detectionId}:`, error.message);

    // Update detection status to failed
    const detection = await Detection.findById(detectionId);
    if (detection) {
      detection.status = "failed";
      detection.error = error.message;
      await detection.save();
    }
  }
};

// ================== SAVE ROAD PARAMETERS ==================
export const saveParameters = async (req, res) => {
  try {
    const { id } = req.params;
    const { roadParameters } = req.body;

    // Validate parameters
    const validation = validateRoadParameters(roadParameters);
    if (!validation.valid) {
      return res.status(400).json({
        success: false,
        message: "Parameter tidak valid",
        errors: validation.errors,
      });
    }

    const detection = await Detection.findByIdAndUpdate(
      id,
      {
        roadParameters: {
          roadName: roadParameters.roadName || "MBZ",
          roadType: roadParameters.roadType || "4/2 D",
          numLanes: roadParameters.numLanes || 4,
          baseCapacity: roadParameters.baseCapacity || 5000,
          laneWidth: roadParameters.laneWidth || 3.5,
          baseSpeed: roadParameters.baseSpeed || 88,
          effectiveWidthFactor: roadParameters.effectiveWidthFactor || 1.0,
        },
      },
      { new: true, runValidators: true }
    );

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    res.json({
      success: true,
      message: "Parameter jalan berhasil disimpan",
      data: detection,
    });
  } catch (error) {
    console.error("Save parameters error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal menyimpan parameter",
      error: error.message,
    });
  }
};

// ================== CALCULATE RESULTS ==================
// Formula: C = n × C0 × FCLE
// DJ = Q / C
// LOS based on DJ
export const calculateResults = async (req, res) => {
  try {
    const { id } = req.params;

    const detection = await Detection.findById(id);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Validate YOLO results
    const yoloValidation = validateYOLOResults(detection.yoloResults);
    if (!yoloValidation.valid) {
      return res.status(400).json({
        success: false,
        message: "Data YOLO tidak lengkap",
        errors: yoloValidation.errors,
      });
    }

    // Validate road parameters
    const paramValidation = validateRoadParameters(detection.roadParameters);
    if (!paramValidation.valid) {
      return res.status(400).json({
        success: false,
        message: "Parameter jalan tidak lengkap",
        errors: paramValidation.errors,
      });
    }

    // Perform calculation
    const calculationResults = performCalculation({
      yoloResults: detection.yoloResults,
      roadParameters: detection.roadParameters,
      videoDuration: detection.videoDuration,
      recordingInterval: detection.recordingInterval,
      roadName: detection.roadParameters.roadName,
    });

    // Update detection with results
    detection.calculations = {
      totalSMP: calculationResults.totalSMP,
      capacity: calculationResults.capacity,
      formula: calculationResults.formula,
      volume: calculationResults.volume,
      degree: calculationResults.degree,
      degreeFormula: calculationResults.degreeFormula,
      los: calculationResults.los,
      losCategory: calculationResults.losCategory,
      losDescription: calculationResults.losDescription,
    };

    detection.conclusion = calculationResults.conclusion;
    detection.status = "completed";
    await detection.save();

    res.json({
      success: true,
      message: "Perhitungan berhasil dilakukan",
      data: {
        capacity: calculationResults.capacity,
        volume: calculationResults.volume,
        degree: calculationResults.degree,
        los: calculationResults.los,
        losCategory: calculationResults.losCategory,
        losDescription: calculationResults.losDescription,
        conclusion: calculationResults.conclusion,
        formula: calculationResults.formula,
      },
    });
  } catch (error) {
    console.error("Calculate results error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal menghitung hasil",
      error: error.message,
    });
  }
};

// ================== GET DETECTION RESULTS ==================
export const getDetectionResults = async (req, res) => {
  try {
    const { id } = req.params;

    console.log(`[${new Date().toISOString()}] Retrieving detection results for ID: ${id}`);

    const detection = await Detection.findById(id).populate("userId", "name email");

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Check ownership or admin
    if (
      detection.userId._id.toString() !== req.user._id.toString() &&
      req.user.role !== "admin"
    ) {
      return res.status(403).json({
        success: false,
        message: "Anda tidak memiliki akses ke detection ini",
      });
    }

    // Set response headers for large data transmission
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Keep-Alive', 'timeout=600, max=100');
    
    // Log transmission start
    console.log(`[${new Date().toISOString()}] Sending detection results, status: ${detection.status}, vehicles: ${detection.yoloResults?.totalVehicles || 0}`);

    // Send response with error handling
    res.json({
      success: true,
      data: detection,
      transmissionInfo: {
        sentAt: new Date().toISOString(),
        dataSize: JSON.stringify(detection).length,
      }
    });

    console.log(`[${new Date().toISOString()}] Detection results sent successfully for ID: ${id}`);

  } catch (error) {
    console.error(`[${new Date().toISOString()}] Get detection results error:`, error);
    
    // Differentiate between different error types
    let statusCode = 500;
    let errorMessage = "Server error";
    let errorType = "UNKNOWN";

    if (error.message.includes("SSL") || error.message.includes("ssl")) {
      statusCode = 502;
      errorMessage = "SSL transmission error - please retry";
      errorType = "SSL_ERROR";
    } else if (error.message.includes("timeout") || error.message.includes("Timeout")) {
      statusCode = 504;
      errorMessage = "Request timeout - video processing may still be in progress";
      errorType = "TIMEOUT_ERROR";
    } else if (error.name === "CastError") {
      statusCode = 400;
      errorMessage = "Invalid detection ID format";
      errorType = "CAST_ERROR";
    }

    res.status(statusCode).json({
      success: false,
      message: errorMessage,
      error: error.message,
      errorType: errorType,
      timestamp: new Date().toISOString(),
    });
  }
};

// ================== GET DETECTION HISTORY ==================
export const getDetectionHistory = async (req, res) => {
  try {
    const { page = 1, limit = 10, status } = req.query;

    const query = {};

    // Admin bisa lihat semua, user/surveyor hanya punya mereka sendiri
    if (req.user.role !== "admin") {
      query.userId = req.user._id;
    }

    if (status) {
      query.status = status;
    }

    const pageNum = parseInt(page);
    const limitNum = parseInt(limit);
    const skip = (pageNum - 1) * limitNum;

    const detections = await Detection.find(query)
      .populate("userId", "name email")
      .skip(skip)
      .limit(limitNum)
      .sort({ createdAt: -1 });

    const total = await Detection.countDocuments(query);

    res.json({
      success: true,
      data: detections,
      pagination: {
        total,
        pages: Math.ceil(total / limitNum),
        currentPage: pageNum,
        limit: limitNum,
      },
    });
  } catch (error) {
    console.error("Get detection history error:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// ================== UPDATE DETECTION STATUS ==================
export const updateDetectionStatus = async (req, res) => {
  try {
    const { id } = req.params;
    const { status, notes } = req.body;

    // Only admin can verify
    if (status === "verified" && req.user.role !== "admin") {
      return res.status(403).json({
        success: false,
        message: "Hanya admin yang dapat memverifikasi",
      });
    }

    const updateData = { status };

    if (status === "verified") {
      updateData.verifiedAt = new Date();
      updateData.verifiedBy = req.user._id;
    }

    if (notes) {
      updateData.notes = notes;
    }

    const detection = await Detection.findByIdAndUpdate(id, updateData, {
      new: true,
    });

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    res.json({
      success: true,
      message: "Status berhasil diperbarui",
      data: detection,
    });
  } catch (error) {
    console.error("Update status error:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// ================== DELETE DETECTION ==================
export const deleteDetection = async (req, res) => {
  try {
    const { id } = req.params;

    const detection = await Detection.findById(id);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Check ownership or admin
    // Allow admin to delete, or owner if userId exists and matches
    if (req.user.role !== "admin") {
      if (!detection.userId || detection.userId.toString() !== req.user._id.toString()) {
        return res.status(403).json({
          success: false,
          message: "Anda tidak memiliki akses ke detection ini",
        });
      }
    }

    // Delete video from Cloudinary
    if (detection.cloudinaryPublicId) {
      try {
        await cloudinary.uploader.destroy(detection.cloudinaryPublicId, {
          resource_type: "video",
        });
      } catch (err) {
        console.error("Error deleting from Cloudinary:", err);
      }
    }

    // Delete detection record
    await Detection.findByIdAndDelete(id);

    res.json({
      success: true,
      message: "Detection berhasil dihapus",
    });
  } catch (error) {
    console.error("Delete detection error:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// ================== GET JOB STATUS (for backward compatibility) ==================
export const getJobStatus = async (req, res) => {
  try {
    const { id } = req.params;

    console.log(`[${new Date().toISOString()}] Checking job status for ID: ${id}`);

    const detection = await Detection.findById(id);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Job ID tidak ditemukan",
      });
    }

    // Calculate progress based on status
    let progress = 0;
    let statusMessage = "";
    
    if (detection.status === "draft") {
      progress = 5;
      statusMessage = "Initializing...";
    } else if (detection.status === "processing") {
      progress = 50;
      statusMessage = "Processing with YOLO...";
    } else if (detection.status === "completed") {
      progress = 100;
      statusMessage = "Complete! Ready for download.";
    } else if (detection.status === "failed") {
      progress = 0;
      statusMessage = `Error: ${detection.error || "Unknown error"}`;
    } else if (detection.status === "verified") {
      progress = 100;
      statusMessage = "Verified! Ready for download.";
    }

    // Set response headers for long-polling stability
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Keep-Alive', 'timeout=600, max=100');

    const statusResponse = {
      success: true,
      id: detection._id,
      status: detection.status,
      progress: progress,
      message: statusMessage,
      yoloResults: detection.yoloResults,
      result: detection.yoloResults,
      calculations: detection.calculations,
      conclusion: detection.conclusion,
      videoUrl: detection.videoUrl,
      outputVideoUrl: detection.yoloResults?.outputVideoUrl || detection.videoUrl,
      createdAt: detection.createdAt,
      completedAt: detection.updatedAt,
      transmission: {
        statusCheckTime: new Date().toISOString(),
        processingDurationSeconds: detection.updatedAt ? 
          Math.floor((detection.updatedAt - detection.createdAt) / 1000) : null,
      }
    };

    console.log(`[${new Date().toISOString()}] Job status response sent - Status: ${detection.status}, Progress: ${progress}%`);

    res.json(statusResponse);

  } catch (error) {
    console.error(`[${new Date().toISOString()}] Get job status error:`, error);
    
    let statusCode = 500;
    let errorMessage = "Error checking job status";
    let errorType = "UNKNOWN";

    if (error.name === "CastError") {
      statusCode = 400;
      errorMessage = "Invalid job ID format";
      errorType = "CAST_ERROR";
    } else if (error.message.includes("connection")) {
      statusCode = 503;
      errorMessage = "Database connection error";
      errorType = "DB_CONNECTION_ERROR";
    }

    res.status(statusCode).json({
      success: false,
      message: errorMessage,
      error: error.message,
      errorType: errorType,
      timestamp: new Date().toISOString(),
    });
  }
};

// ================== SAVE YOLO RESULTS FROM RAILWAY ==================
export const saveYOLOResults = async (req, res) => {
  try {
    const { video_url, output_video_url, video_name, total_vehicles, avg_confidence, duration, frames, detections, summary } = req.body;

    if (!video_url || !video_name) {
      return res.status(400).json({
        success: false,
        message: "Video URL dan nama wajib diisi",
      });
    }

    // Extract cloudinaryPublicId from URL
    // URL format: https://res.cloudinary.com/{cloud}/video/upload/{version}/{public_id}.{ext}
    let cloudinaryPublicId = "unknown";
    try {
      const url = new URL(video_url);
      const pathParts = url.pathname.split("/");
      const uploadIndex = pathParts.indexOf("upload");
      if (uploadIndex !== -1 && uploadIndex < pathParts.length - 1) {
        // Get everything after /upload/{version}/ as public_id (remove extension)
        const fileWithExt = pathParts.slice(uploadIndex + 2).join("/");
        cloudinaryPublicId = fileWithExt.split(".")[0]; // Remove file extension
      }
    } catch (err) {
      console.warn("⚠️ Could not extract cloudinaryPublicId from URL:", err.message);
    }

    // Extract vehicle types from detections array
    // Detections format: Array of {class: "mobil|bus|truk", ...}
    let vehicleTypes = {
      mobil: 0,
      bus: 0,
      truk: 0,
    };

    let totalFrames = parseInt(frames) || 0;

    if (Array.isArray(detections) && detections.length > 0) {
      // If detections is an array with vehicle type data (from YOLO result)
      detections.forEach((item) => {
        if (item.type === "Car" || item.class === "mobil") vehicleTypes.mobil += (item.count || 1);
        else if (item.type === "Bus" || item.class === "bus") vehicleTypes.bus += (item.count || 1);
        else if (item.type === "Truck" || item.class === "truk") {
          // Combine all truck types into one
          vehicleTypes.truk += (item.count || 1);
        }
      });
    }

    // NEW: Extract per-lane vehicle breakdown from summary
    const perLaneSummary = summary || {};
    const leftLaneVehicles = perLaneSummary.leftLane || { mobil: 0, bus: 0, truk: 0 };
    const rightLaneVehicles = perLaneSummary.rightLane || { mobil: 0, bus: 0, truk: 0 };

    // Create new detection record
    const detection = new Detection({
      userId: req.user._id,
      videoUrl: video_url,
      outputVideoUrl: output_video_url,
      cloudinarySecureUrl: video_url,
      cloudinaryPublicId: cloudinaryPublicId,
      fileName: video_name,
      videoDuration: parseInt(duration) || 0,
      status: "draft",
      yoloResults: {
        totalVehicles: total_vehicles || 0,
        avgConfidence: avg_confidence || 0,
        vehicleTypes: vehicleTypes,
        totalFrames: totalFrames,
        rawData: detections || [],
        // NEW: Per-lane vehicle breakdown
        leftLaneCount: perLaneSummary.leftLaneCount || 0,
        rightLaneCount: perLaneSummary.rightLaneCount || 0,
        leftLane: {
          mobil: leftLaneVehicles.mobil || 0,
          bus: leftLaneVehicles.bus || 0,
          truk: leftLaneVehicles.truk || 0,
        },
        rightLane: {
          mobil: rightLaneVehicles.mobil || 0,
          bus: rightLaneVehicles.bus || 0,
          truk: rightLaneVehicles.truk || 0,
        },
      },
    });

    await detection.save();

    res.status(201).json({
      success: true,
      message: "Data YOLO tersimpan berhasil",
      detection: {
        _id: detection._id,
        videoName: detection.fileName,
        totalVehicles: detection.yoloResults.totalVehicles,
        avgConfidence: detection.yoloResults.avgConfidence,
        // NEW: Return per-lane data
        leftLaneCount: detection.yoloResults.leftLaneCount,
        rightLaneCount: detection.yoloResults.rightLaneCount,
        leftLane: detection.yoloResults.leftLane,
        rightLane: detection.yoloResults.rightLane,
        status: detection.status,
      },
    });
  } catch (error) {
    console.error("Save YOLO results error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal menyimpan hasil YOLO",
      error: error.message,
    });
  }
};
