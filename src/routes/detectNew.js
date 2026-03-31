/**
 * YOLO Detection Routes - LOCAL PROCESSING (No Railway)
 * Uses count_video.py in backend/yolo/ for reliable processing
 */

import express from "express";
import axios from "axios";
import FormData from "form-data";
import jwt from "jsonwebtoken";
import fs from "fs";
import path from "path";
import upload from "../middlewares/uploadVideo.js";
import { jobQueue } from "../utils/jobQueue.js";
import { runCountVideo, checkPythonSetup } from "../utils/pythonRunner.js";
import {
  uploadToYoloAPI,
  pollYoloResult,
  processVideoLocally,
  parseYoloResult,
  downloadFromCloudinary,
  validateVideoFile,
  processVideoFromCloudinary,
} from "../services/yoloService.js";

// Import existing controllers
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

// ===== INITIALIZATION =====

// Simple middleware for file upload to local storage
function getUploadMiddleware() {
  return upload.single("video");
}

const router = express.Router();

// ===== MIDDLEWARE =====

const verifyToken = (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      return res.status(401).json({ error: "No authorization header" });
    }
    
    const parts = authHeader.split(" ");
    if (parts.length !== 2 || parts[0] !== "Bearer") {
      return res.status(401).json({ error: "Invalid auth format. Use: Bearer <token>" });
    }
    
    const token = parts[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = decoded.id;
    next();
  } catch (err) {
    console.error("❌ [Auth] Verification failed:", err.message);
    return res.status(401).json({ error: "Invalid token", details: err.message });
  }
};

// ===== ENDPOINTS =====

/**
 * POST /api/detect/yolo/start-job
 * Start new YOLO processing job (upload + process)
 * Returns job ID untuk tracking progress
 */
router.post("/yolo/start-job", verifyToken, (req, res, next) => getUploadMiddleware()(req, res, next), async (req, res) => {
  console.log("\n========================================");
  console.log("🎬 [Job] POST /yolo/start-job received");
  console.log("========================================");
  
  try {
    console.log("✅ Step 1: Auth verified, userId =", req.userId);
    console.log("✅ Step 2: File uploaded?", req.file ? "YES" : "NO");
    
    if (!req.file) {
      console.log("❌ Step 2: No file in request");
      return res.status(400).json({ 
        error: "No video file uploaded",
        received: {
          body: req.body ? Object.keys(req.body) : [],
          file: req.file ? "present" : "missing",
        }
      });
    }

    const fileName = req.file.filename || req.file.originalname;
    const cloudinaryUrl = req.file.path;
    
    console.log("✅ Step 3: Extract file info");
    console.log("   File name:", fileName);
    console.log("   Cloudinary URL:", cloudinaryUrl ? cloudinaryUrl.substring(0, 50) + "..." : "MISSING");

    // Create job untuk tracking
    console.log("⏳ Step 4: Creating job in queue...");
    let jobId;
    try {
      jobId = jobQueue.createJob('yolo_processing', {
        videoName: fileName,
        cloudinaryUrl: cloudinaryUrl,
        userId: req.userId,
        startedAt: new Date(),
      });
      console.log("✅ Step 4: Job created, jobId =", jobId);
    } catch (queueErr) {
      console.error("❌ Step 4: Failed to create job:", queueErr.message);
      throw queueErr;
    }

    // Start processing async (jangan await)
    console.log("⏳ Step 5: Starting async YOLO processing...");
    let processingStarted = false;
    try {
      processYOLOJob(jobId, cloudinaryUrl, fileName).catch(err => {
        console.error(`❌ [Job ${jobId}] Processing error:`, err);
        try {
          jobQueue.failJob(jobId, err);
          console.log(`✅ [Job ${jobId}] Marked as failed in queue`);
        } catch (failErr) {
          console.error(`❌ [Job ${jobId}] Failed to mark as failed:`, failErr.message);
        }
      });
      processingStarted = true;
      console.log("✅ Step 5: Async processing started successfully");
    } catch (processStartErr) {
      console.error("❌ Step 5: Failed to start processing:", processStartErr.message);
      throw processStartErr;
    }

    // Return job ID immediately ke frontend
    console.log("⏳ Step 6: Returning response to client...");
    const response = {
      success: true,
      jobId: jobId,
      message: "Job started. Use jobId to track progress.",
      cloudinaryUrl: cloudinaryUrl,
    };
    console.log("✅ Step 6: Response prepared:", response);
    
    return res.json(response);
    
  } catch (err) {
    console.error("\n❌ [Job] ROUTE HANDLER ERROR:");
    console.error("   Message:", err.message);
    console.error("   Name:", err.name);
    console.error("   Stack:", err.stack);
    console.error("========================================\n");
    
    return res.status(500).json({
      error: "Failed to start job",
      details: err.message,
      errorName: err.name,
    });
  }
});

/**
 * POST /api/detect/yolo/create-job
 * Create job from Cloudinary URL (called by frontend after direct upload)
 */
router.post("/yolo/create-job", verifyToken, async (req, res) => {
  console.log("\n========================================");
  console.log("🎬 [Job] POST /yolo/create-job received");
  console.log("========================================");
  
  try {
    const { cloudinaryUrl, videoName } = req.body;
    
    console.log("✅ Step 1: Auth verified, userId =", req.userId);
    console.log("✅ Step 2: Request body:", { cloudinaryUrl: cloudinaryUrl ? cloudinaryUrl.substring(0, 50) + "..." : "MISSING", videoName });
    
    if (!cloudinaryUrl) {
      return res.status(400).json({ success: false, error: "Missing cloudinaryUrl" });
    }

    // Validate jobQueue exists
    if (!jobQueue || typeof jobQueue.createJob !== 'function') {
      console.error("❌ [Job] jobQueue not available or not properly initialized");
      return res.status(500).json({ 
        success: false, 
        error: "Job queue service unavailable",
        details: "jobQueue is not initialized"
      });
    }
    
    // Create job untuk tracking
    console.log("⏳ Step 3: Creating job in queue...");
    let jobId;
    try {
      jobId = jobQueue.createJob('yolo_processing', {
        videoName: videoName || 'video',
        cloudinaryUrl: cloudinaryUrl,
        userId: req.userId,
        startedAt: new Date(),
      });
    } catch (qErr) {
      console.error("❌ [Job] Failed to create job in queue:", qErr.message, qErr.stack);
      return res.status(500).json({
        success: false,
        error: "Failed to create job in queue",
        details: qErr.message
      });
    }
    
    console.log("✅ Step 3: Job created, jobId =", jobId);
    console.log("✅ About to start async processing...");
    
    // Start processing async (jangan await)
    (async () => {
      try {
        console.log("🚀 [Job] Starting YOLO processing for:", jobId);
        jobQueue.startJob(jobId);
        
        console.log("🔍 [Job] Calling processVideoFromCloudinary...");
        const result = await processVideoFromCloudinary(cloudinaryUrl, {
          jobId: jobId,
          onProgress: (progress, message) => {
            if (jobQueue && typeof jobQueue.updateProgress === 'function') {
              jobQueue.updateProgress(jobId, progress, message);
            }
          }
        });
        
        console.log("✅ [Job] YOLO processing complete:", jobId);
        if (jobQueue && typeof jobQueue.completeJob === 'function') {
          jobQueue.completeJob(jobId, result);
        }
      } catch (err) {
        console.error("❌ [Job] YOLO processing failed:", err.message, err.stack);
        if (jobQueue && typeof jobQueue.failJob === 'function') {
          jobQueue.failJob(jobId, err.message);
        }
      }
    })();
    
    return res.json({
      success: true,
      jobId: jobId,
      message: "Job created and processing started",
      cloudinaryUrl: cloudinaryUrl,
    });
  } catch (err) {
    console.error("❌ [Job] Create job error:", err.message);
    console.error("   Stack:", err.stack);
    console.error("   Full error:", JSON.stringify(err, null, 2));
    
    // Make sure response hasn't been sent yet
    if (!res.headersSent) {
      return res.status(500).json({
        success: false,
        error: "Failed to create job",
        details: err.message,
        code: err.code || "UNKNOWN"
      });
    } else {
      console.error("❌ Headers already sent, cannot respond");
    }
  }
});

/**
 * GET /api/detect/yolo/job/:jobId
 * Get job status dan progress
 * NO AUTH REQUIRED - polling endpoint can be called without token
 */
router.get("/yolo/job/:jobId", (req, res) => {
  try {
    const jobId = req.params.jobId;
    const job = jobQueue.getJob(jobId);

    if (!job) {
      return res.status(404).json({
        error: "Job not found",
        jobId: jobId,
      });
    }

    return res.json({
      jobId: job.id,
      status: job.status, // pending, processing, completed, failed
      progress: job.progress, // 0-100
      message: job.message || '',
      result: job.status === 'completed' ? job.result : null,
      error: job.status === 'failed' ? job.error : null,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
    });
  } catch (err) {
    console.error("❌ [Job] Get status error:", err.message);
    return res.status(500).json({
      error: "Failed to get job status",
      details: err.message,
    });
  }
});

/**
 * GET /api/detect/yolo/job/:jobId/stream
 * Server-Sent Events (SSE) untuk real-time progress streaming
 * NO AUTH REQUIRED - polling endpoint can be called without token
 */
router.get("/yolo/job/:jobId/stream", (req, res) => {
  try {
    const jobId = req.params.jobId;
    const job = jobQueue.getJob(jobId);

    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }

    // Setup SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Send initial status
    sendSSEMessage(res, 'job_status', {
      jobId: job.id,
      status: job.status,
      progress: job.progress,
    });

    // Poll job status setiap 1 detik
    let lastProgress = job.progress;
    const pollInterval = setInterval(() => {
      const currentJob = jobQueue.getJob(jobId);
      
      if (!currentJob) {
        sendSSEMessage(res, 'error', { message: 'Job not found' });
        clearInterval(pollInterval);
        res.end();
        return;
      }

      // Send update jika ada perubahan
      if (currentJob.progress !== lastProgress || currentJob.status !== 'processing') {
        sendSSEMessage(res, 'job_update', {
          status: currentJob.status,
          progress: currentJob.progress,
          message: currentJob.message || '',
        });
        lastProgress = currentJob.progress;
      }

      // Stop jika job selesai
      if (currentJob.status !== 'processing') {
        sendSSEMessage(res, 'job_complete', {
          status: currentJob.status,
          progress: currentJob.progress,
          result: currentJob.status === 'completed' ? currentJob.result : null,
          error: currentJob.status === 'failed' ? currentJob.error : null,
        });
        clearInterval(pollInterval);
        res.end();
      }
    }, 1000);

    // Cleanup jika client disconnect
    req.on('close', () => {
      clearInterval(pollInterval);
      res.end();
    });

  } catch (err) {
    console.error("❌ [SSE] Stream error:", err.message);
    return res.status(500).json({ error: "Failed to stream job updates" });
  }
});

/**
 * Background async function untuk process YOLO
 * UPDATED: Langsung menggunakan local count_video.py
 * NO RAILWAY DEPENDENCY
 */
async function processYOLOJob(jobId, cloudinaryUrl, fileName) {
  const job = jobQueue.startJob(jobId);
  
  try {
    console.log(`\n🚀 [${jobId}] Starting local YOLO processing`);
    console.log(`📁 Processing method: LOCAL (count_video.py)`);
    
    // ===== STEP 1: Download dari Cloudinary =====
    jobQueue.updateProgress(jobId, 10, 'Downloading video from Cloudinary...');
    console.log(`[${jobId}] Step 1: Download from Cloudinary`);
    
    let videoPath;
    let jobWorkDir;
    try {
      // 🔧 FIX 1: Use /tmp for Vercel compatibility (no mkdir on serverless)
      jobWorkDir = path.join('/tmp', `yolo_job_${jobId}`);
      console.log(`📁 Using job directory: ${jobWorkDir}`);
      
      // 🔧 FIX 2: Sanitize filename and ensure .mp4 extension
      let safeFileName = fileName || 'video.mp4';
      // Remove any path separators for security
      safeFileName = path.basename(safeFileName);
      // Ensure .mp4 extension
      if (!safeFileName.toLowerCase().endsWith('.mp4')) {
        safeFileName = `${path.parse(safeFileName).name}.mp4`;
      }
      
      videoPath = path.join(jobWorkDir, safeFileName);
      console.log(`📥 Target path: ${videoPath}`);
      
      // Download with error handling
      const response = await axios.get(cloudinaryUrl, {
        responseType: 'arraybuffer',
        timeout: 120000,
      });
      
      // 🔧 FIX 3: Validate download size before write
      if (!response.data || response.data.length === 0) {
        throw new Error('Downloaded file is empty');
      }
      
      console.log(`💾 Writing ${response.data.length} bytes to ${videoPath}`);
      fs.writeFileSync(videoPath, response.data);
      
      // 🔧 FIX 4: Validate file exists after write
      if (!fs.existsSync(videoPath)) {
        throw new Error(`File write failed - file not found at ${videoPath}`);
      }
      
      const fileStats = fs.statSync(videoPath);
      if (fileStats.size === 0) {
        throw new Error(`File written but size is 0 at ${videoPath}`);
      }
      
      console.log(`✅ [${jobId}] Downloaded: ${videoPath} (${fileStats.size} bytes)`);
      
    } catch (downloadErr) {
      throw new Error(`Download failed: ${downloadErr.message}`);
    }

    // ===== STEP 2: Process dengan LOCAL count_video.py =====
    console.log(`[${jobId}] Step 2: Process with local count_video.py`);
    jobQueue.updateProgress(jobId, 15, 'Starting local YOLO processing...');
    
    let yoloResult;
    try {
      // Process locally dengan progress callback
      yoloResult = await processVideoLocally(videoPath, (progress) => {
        // Map progress: 15% (start) → 90% (end)
        const mappedProgress = Math.min(90, progress);
        jobQueue.updateProgress(jobId, mappedProgress, 
          `Processing video: ${progress}%`);
      });
      
      console.log(`✅ [${jobId}] Processing completed`);
      console.log(`📊 [${jobId}] Result:`, yoloResult);
      
    } catch (processErr) {
      console.error(`❌ [${jobId}] Processing error:`, processErr.message);
      throw new Error(`Local processing failed: ${processErr.message}`);
    } finally {
      // Cleanup: Move output video to permanent location, then delete temp directory
      try {
        // 🔧 FIX 5: Skip permanent storage on Vercel/serverless (ephemeral filesystem)
        if (yoloResult && yoloResult.outputVideoPath && fs.existsSync(yoloResult.outputVideoPath)) {
          // Detect if we're running on serverless
          const isServerless = () => {
            return process.env.VERCEL_RUNTIME || 
                   process.env.NOW_REGION || 
                   process.env.LAMBDA_TASK_ROOT ||
                   /var\/task/.test(process.cwd());
          };
          // On Vercel/serverless, just keep in temp; on local, try to move to uploads
          if (!isServerless()) {
            const uploadsDir = path.join(process.cwd(), '..', 'backend', 'uploads');
            try {
              if (!fs.existsSync(uploadsDir)) {
                fs.mkdirSync(uploadsDir, { recursive: true });
              }
            } catch (mkErr) {
              // Silent fail - just skip mkdir on read-only FS
            }
            
            const permanentPath = path.join(uploadsDir, `${jobId}_detected.mp4`);
            fs.copyFileSync(yoloResult.outputVideoPath, permanentPath);
            console.log(`📦 [${jobId}] Moved output video to: ${permanentPath}`);
            
            // Update the path in result to point to permanent location
            yoloResult.outputVideoPath = permanentPath;
          }
        }
        
        // Delete input video
        if (videoPath && fs.existsSync(videoPath)) {
          fs.unlinkSync(videoPath);
          console.log(`🧹 [${jobId}] Cleaned up temp input file`);
        }
        
        // Remove entire job directory
        if (jobWorkDir && fs.existsSync(jobWorkDir)) {
          const files = fs.readdirSync(jobWorkDir);
          for (const file of files) {
            const filePath = path.join(jobWorkDir, file);
            if (fs.lstatSync(filePath).isDirectory()) {
              fs.rmdirSync(filePath);
            } else {
              fs.unlinkSync(filePath);
            }
          }
          fs.rmdirSync(jobWorkDir);
          console.log(`🧹 [${jobId}] Cleaned up job directory`);
        }
      } catch (cleanupErr) {
        console.warn(`⚠️ [${jobId}] Cleanup warning:`, cleanupErr.message);
      }
    }

    // ===== STEP 3: Parse results =====
    jobQueue.updateProgress(jobId, 90, 'Parsing results...');
    const summary = parseYoloResult(yoloResult);
    console.log(`📊 [${jobId}] Summary:`, summary);
    
    // ===== STEP 3.5: Upload output video if exists =====
    let outputVideoUrl = null;
    if (yoloResult.outputVideoPath && fs.existsSync(yoloResult.outputVideoPath)) {
      try {
        console.log(`📤 [${jobId}] Uploading processed video to Cloudinary...`);
        jobQueue.updateProgress(jobId, 95, 'Uploading processed video...');
        
        const fileStream = fs.createReadStream(yoloResult.outputVideoPath);
        const result = await new Promise((resolve, reject) => {
          // ⭐ FIX: Increase timeout to 30 minutes for large video uploads
          const timeoutMs = 1800000; // 30 minutes
          let uploadStream = null;
          let isResolved = false;
          
          // Set timeout handler
          const timeoutHandle = setTimeout(() => {
            if (!isResolved) {
              isResolved = true;
              if (uploadStream) {
                uploadStream.destroy();
              }
              reject(new Error(`Upload timeout exceeded after ${timeoutMs}ms`));
            }
          }, timeoutMs);
          
          uploadStream = cloudinary.uploader.upload_stream(
            {
              resource_type: 'video',
              folder: 'detection_videos_output',
              public_id: `${fileName.replace(/\.[^/.]+$/, '')}_detected`,
              timeout: timeoutMs,
            },
            (error, result) => {
              clearTimeout(timeoutHandle);
              if (!isResolved) {
                isResolved = true;
                if (error) reject(error);
                else resolve(result);
              }
            }
          );
          
          fileStream.on('error', (err) => {
            clearTimeout(timeoutHandle);
            if (!isResolved) {
              isResolved = true;
              reject(new Error(`File stream error: ${err.message}`));
            }
          });
          
          fileStream.pipe(uploadStream);
        });
        
        outputVideoUrl = result.secure_url;
        console.log(`✅ [${jobId}] Output video uploaded: ${outputVideoUrl}`);
      } catch (uploadErr) {
        console.warn(`⚠️ [${jobId}] Failed to upload output video:`, uploadErr.message);
        // Don't fail the whole job, just continue without output video
      }
    }

    // ===== STEP 4: Complete =====
    jobQueue.updateProgress(jobId, 100, 'Processing complete!');
    console.log(`✅ [${jobId}] Complete with YOLO result:`, yoloResult);
    console.log(`✅ [${jobId}] Parsed summary:`, summary);

    jobQueue.completeJob(jobId, {
      // Return the PARSED summary (with corrected totalVehicles) instead of raw yoloResult
      // This ensures totalVehicles = leftLaneCount + rightLaneCount, not the raw API value
      ...summary,
      videoName: fileName,
      cloudinaryUrl: cloudinaryUrl,
      outputVideoUrl: outputVideoUrl,
    });

  } catch (err) {
    console.error(`❌ [${jobId}] Processing failed:`, err.message);
    
    // Cleanup on error
    try {
      if (jobWorkDir && fs.existsSync(jobWorkDir)) {
        const files = fs.readdirSync(jobWorkDir);
        files.forEach(file => {
          fs.unlinkSync(path.join(jobWorkDir, file));
        });
        fs.rmdirSync(jobWorkDir);
        console.log(`🧹 [${jobId}] Cleaned up job directory after error`);
      }
    } catch (cleanupErr) {
      console.warn(`⚠️ [${jobId}] Cleanup error:`, cleanupErr.message);
    }
    
    jobQueue.failJob(jobId, err.message);
  }
}

// ===== HELPER FUNCTIONS =====

// ===== HELPER FUNCTIONS =====

function sendSSEMessage(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ===== FALLBACK MOCK ENDPOINT =====

router.post("/yolo/mock", verifyToken, (req, res) => {
  const jobId = jobQueue.createJob('mock_yolo', { mode: 'mock' });
  jobQueue.completeJob(jobId, {
    videoName: "mock_video.mp4",
    summary: {
      totalVehicles: 1247,
      carCount: 998,
      truckCount: 187,
      busCount: 62,
      leftLaneCount: 623,
      rightLaneCount: 624,
      confidence: '0.87',
    },
  });

  return res.json({
    success: true,
    jobId: jobId,
    message: "Mock job created",
  });
});

// ===== CHUNKED UPLOAD MIDDLEWARE =====
function getChunkedUploadMiddleware() {
  return (req, res, next) => {
    // Detect if we're running on serverless
    const isServerless = () => {
      return process.env.VERCEL_RUNTIME || 
             process.env.NOW_REGION || 
             process.env.LAMBDA_TASK_ROOT ||
             /var\/task/.test(process.cwd());
    };
    
    // Use /tmp on Vercel/serverless (only writable location), or temp directory locally
    const tempDir = isServerless() ? '/tmp' : path.join(process.cwd(), 'temp');
    
    // Silent mkdir - never throw on read-only filesystem
    if (!isServerless()) {
      try {
        if (!fs.existsSync(tempDir)) {
          fs.mkdirSync(tempDir, { recursive: true });
        }
      } catch (err) {
        // Silent fail - Vercel read-only FS
      }
    }
    
    const storage = multer.diskStorage({
      destination: (req, file, cb) => {
        cb(null, tempDir);
      },
      filename: (req, file, cb) => {
        const chunkIndex = req.body.chunkIndex || 0;
        const uploadSessionId = req.body.uploadSessionId || Date.now();
        cb(null, `chunk_${uploadSessionId}_${chunkIndex}_${Date.now()}`);
      }
    });
    
    const chunkUpload = multer({
      storage,
      limits: { fileSize: 4 * 1024 * 1024 } // 4MB per chunk (stays under Vercel's 4.5MB limit)
    });
    
    chunkUpload.single("chunk")(req, res, function(err) {
      if (err) {
        console.error("❌ Chunk upload error:", err.message);
        return res.status(400).json({
          error: "Chunk upload failed",
          details: err.message
        });
      }
      next();
    });
  };
}

// ===== CHUNKED UPLOAD HANDLER =====
// For handling large videos in chunks (Vercel 4.5MB limit workaround)
const uploadSessions = new Map(); // Store upload sessions: uploadSessionId -> {chunks, totalChunks, fileName, tempPath}

router.post("/yolo/upload-chunk", protect, (req, res, next) => getChunkedUploadMiddleware()(req, res, next), async (req, res) => {
  try {
    const { chunkIndex, totalChunks, fileName, uploadSessionId } = req.body;
    
    if (!req.file) {
      return res.status(400).json({ error: "No chunk file provided" });
    }
    
    if (chunkIndex === undefined || totalChunks === undefined) {
      return res.status(400).json({ error: "Missing chunkIndex or totalChunks" });
    }
    
    console.log(`📦 Received chunk ${parseInt(chunkIndex) + 1}/${totalChunks} (${req.file.size} bytes)`);
    
    let sessionId = uploadSessionId;
    if (!sessionId) {
      sessionId = `upload_${Date.now()}_${Math.random().toString(36).substring(7)}`;
      uploadSessions.set(sessionId, {
        chunks: {},
        totalChunks: parseInt(totalChunks),
        fileName: fileName,
        userId: req.user.id
      });
      console.log(`🆕 Created upload session: ${sessionId}`);
    }
    
    const session = uploadSessions.get(sessionId);
    if (!session) {
      return res.status(400).json({ error: "Invalid upload session" });
    }
    
    // Store this chunk
    session.chunks[parseInt(chunkIndex)] = req.file.path;
    const receivedChunks = Object.keys(session.chunks).length;
    
    console.log(`✅ Chunk ${parseInt(chunkIndex) + 1} stored. Progress: ${receivedChunks}/${session.totalChunks}`);
    
    // Check if all chunks received
    if (receivedChunks === session.totalChunks) {
      console.log(`🔗 All chunks received! Assembling file...`);
      
      try {
        // Detect if we're running on serverless
        const isServerless = () => {
          return process.env.VERCEL_RUNTIME || 
                 process.env.NOW_REGION || 
                 process.env.LAMBDA_TASK_ROOT ||
                 /var\/task/.test(process.cwd()) ||
                 process.cwd().includes('var/task');
        };
        
        // Use /tmp on Vercel, or temp directory locally
        const tempDir = isServerless() ? '/tmp' : path.join(process.cwd(), 'temp');
        const reconstructed = path.join(tempDir, `combined_${Date.now()}.mp4`);
        const writeStream = fs.createWriteStream(reconstructed);
        
        for (let i = 0; i < session.totalChunks; i++) {
          const chunkPath = session.chunks[i];
          const chunkData = fs.readFileSync(chunkPath);
          writeStream.write(chunkData);
          fs.unlinkSync(chunkPath); // Clean up chunk file
        }
        writeStream.end();
        
        await new Promise((resolve, reject) => {
          writeStream.on('finish', resolve);
          writeStream.on('error', reject);
        });
        
        console.log(`✅ File reconstructed: ${reconstructed}`);
        
        // Now upload reconstructed file to Cloudinary
        console.log("📤 Uploading reconstructed file to Cloudinary...");
        const stream = fs.createReadStream(reconstructed);
        
        const uploadPromise = new Promise((resolve, reject) => {
          const upload_stream = cloudinary.uploader.upload_stream(
            {
              resource_type: "video",
              folder: "pktj_videos"
            },
            (error, result) => {
              if (error) reject(error);
              else resolve(result);
            }
          );
          stream.pipe(upload_stream);
        });
        
        const cloudinaryResult = await uploadPromise;
        const cloudinaryUrl = cloudinaryResult.secure_url;
        
        console.log("✅ File uploaded to Cloudinary:", cloudinaryUrl.substring(0, 50) + "...");
        
        // Create job
        console.log("⏳ Creating YOLO job...");
        const jobId = jobQueue.createJob('yolo_processing', {
          videoName: session.fileName,
          cloudinaryUrl: cloudinaryUrl,
          userId: req.user.id,
        });
        
        console.log(`✅ Job created: ${jobId}`);
        
        // Clean up session and temp file
        uploadSessions.delete(sessionId);
        fs.unlinkSync(reconstructed);
        
        // Return success with jobId
        return res.json({
          success: true,
          uploadSessionId: sessionId,
          jobId: jobId,
          cloudinaryUrl: cloudinaryUrl,
          message: "File assembled and job created"
        });
        
      } catch (assemblyErr) {
        console.error("❌ File assembly error:", assemblyErr.message);
        uploadSessions.delete(sessionId);
        return res.status(500).json({
          error: "File assembly failed",
          details: assemblyErr.message
        });
      }
    }
    
    // More chunks expected
    return res.json({
      success: true,
      uploadSessionId: sessionId,
      receivedChunks: receivedChunks,
      totalChunks: session.totalChunks,
      message: `Chunk ${parseInt(chunkIndex) + 1} received, waiting for remaining chunks`
    });
    
  } catch (err) {
    console.error("❌ Chunk upload error:", err.message);
    res.status(500).json({
      error: "Chunk upload failed",
      details: err.message
    });
  }
});

// ===== ORIGINAL ROUTES =====
// NOTE: Non-parameterized routes MUST come before parameterized routes (/:id/...)
// Otherwise Express matches specific routes against the :id parameter

// Non-parameterized routes (must be first!)
router.post("/", protect, saveYOLOResults);
router.post("/upload", protect, getUploadMiddleware(), uploadVideo);
router.post("/process", protect, processVideo);
router.get("/history", protect, getDetectionHistory);
router.get("/results/:id", getDetectionResults);

// ===== DOWNLOAD VIDEO ROUTES =====
// GET /api/detect/download/output/:detectionId - Download output video
router.get("/download/output/:detectionId", protect, async (req, res) => {
  try {
    const { detectionId } = req.params;
    const detection = await require("../models/Detection.js").default.findById(detectionId);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Check authorization
    if (detection.userId.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: "Anda tidak memiliki akses untuk video ini",
      });
    }

    const videoPath = detection.yoloResults?.outputVideoUrl || detection.videoUrl;
    
    if (!videoPath) {
      return res.status(404).json({
        success: false,
        message: "Output video belum tersedia",
      });
    }

    // If it's a local path, serve it
    if (fs.existsSync(videoPath)) {
      console.log(`📥 [DOWNLOAD] Serving video: ${videoPath}`);
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Content-Disposition', `attachment; filename="${path.basename(videoPath)}"`);
      const fileStream = fs.createReadStream(videoPath);
      fileStream.pipe(res);
    } else {
      // If it's a URL, redirect to it
      console.log(`🔗 [DOWNLOAD] Redirecting to URL: ${videoPath}`);
      return res.json({
        success: true,
        url: videoPath,
        message: "Silakan buka URL di browser untuk download video",
      });
    }
  } catch (error) {
    console.error("❌ [DOWNLOAD] Error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal download video",
      error: error.message,
    });
  }
});

// GET /api/detect/stream/:detectionId - Stream output video
router.get("/stream/:detectionId", protect, async (req, res) => {
  try {
    const { detectionId } = req.params;
    const Detection = require("../models/Detection.js").default;
    const detection = await Detection.findById(detectionId);

    if (!detection) {
      return res.status(404).json({
        success: false,
        message: "Detection tidak ditemukan",
      });
    }

    // Check authorization
    if (detection.userId.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: "Anda tidak memiliki akses untuk video ini",
      });
    }

    const videoPath = detection.yoloResults?.outputVideoUrl || detection.videoUrl;
    
    if (!videoPath || !fs.existsSync(videoPath)) {
      return res.status(404).json({
        success: false,
        message: "Output video tidak tersedia",
      });
    }

    console.log(`📹 [STREAM] Streaming video: ${videoPath}`);
    
    const fileSize = fs.statSync(videoPath).size;
    const range = req.headers.range;

    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": end - start + 1,
        "Content-Type": "video/mp4",
      });
      fs.createReadStream(videoPath, { start, end }).pipe(res);
    } else {
      res.writeHead(200, {
        "Content-Length": fileSize,
        "Content-Type": "video/mp4",
      });
      fs.createReadStream(videoPath).pipe(res);
    }
  } catch (error) {
    console.error("❌ [STREAM] Error:", error);
    res.status(500).json({
      success: false,
      message: "Gagal stream video",
      error: error.message,
    });
  }
});

// Parameterized routes (must be last!)
router.put("/:id/parameters", protect, saveParameters);
router.post("/:id/calculate", protect, calculateResults);
router.patch("/:id/status", protect, updateDetectionStatus);
router.delete("/:id", protect, deleteDetection);
router.get("/:id/status", protect, getJobStatus);

export default router;
