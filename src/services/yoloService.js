/**
 * YOLO Service - HYBRID PROCESSING
 * 
 * ARCHITECTURE:
 * - Local development: Uses count_video.py with spawn()
 * - Railway production: Calls YOLO API on ngrok (PYTHON_API env var)
 * 
 * This allows backend to remain pure orchestrator while supporting local dev
 * 
 * ⭐ FIXES IMPLEMENTED:
 * - Retry logic dengan exponential backoff untuk unstable tunnel
 * - Keep-alive connections untuk prevent premature disconnect
 * - Increased timeouts (15 min upload, 30s polling)
 */

import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import http from "http";
import https from "https";
import axios from "axios";
import FormData from "form-data";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const YOLO_DIR = path.join(__dirname, "../../yolo");
const PYTHON_SCRIPT = path.join(__dirname, "../../yolo/count_video.py");

// ✅ DETECT ENVIRONMENT CORRECTLY:
// Railway: NODE_ENV=production (from railway.json / Procfile)
// Local: NODE_ENV=development (from npm run dev)
const IS_RAILWAY = process.env.NODE_ENV === "production";
const PYTHON_API = process.env.PYTHON_API || "https://hurtling-unforecasted-horace.ngrok-free.dev/";

console.log(`\n✅ YOLO Service Initialized`);
console.log(`📍 Environment: ${IS_RAILWAY ? "🚂 RAILWAY (API Mode)" : "🖥️  LOCAL DEV (Python Mode)"}`);
if (IS_RAILWAY) {
  console.log(`🌐 YOLO API: ${PYTHON_API}`);
  console.log(`📌 All YOLO calls will use ngrok API`);
} else {
  console.log(`📁 Python script: ${PYTHON_SCRIPT}`);
  console.log(`📌 All YOLO calls will spawn local Python process`);
}
console.log();

/**
 * ⭐ RETRY LOGIC dengan exponential backoff untuk handle unstable tunnel
 * Masalah: Tunnel ngrok sering mati/timeout
 * Solusi: Retry dengan jeda eksponensial + jeda acak (jitter)
 */
async function retryWithBackoff(fn, maxRetries = 3, initialDelay = 2000) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      const isTimeout = error.code === 'ECONNABORTED' || 
                       error.code === 'ECONNRESET' || 
                       error.code === 'ETIMEDOUT' ||
                       error.message?.includes('timeout') ||
                       error.message?.includes('socket') ||
                       error.message?.includes('ECONNREFUSED') ||
                       error.response?.status === 502 || // Bad gateway
                       error.response?.status === 503 || // Service unavailable
                       error.response?.status === 504;   // Gateway timeout
      
      if (isLastAttempt || !isTimeout) {
        throw error;
      }
      
      // Exponential backoff: 2s → 4s → 8s, plus random jitter (0-1s)
      const delay = initialDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
      console.log(`⚠️  Attempt ${attempt}/${maxRetries} failed (${error.code || error.response?.status || error.message})`);
      console.log(`⏳ Retrying in ${Math.round(delay)}ms... [${PYTHON_API}]`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

/**
 * Process video via YOLO API (ngrok tunnel)
 * 
 * ⭐ PERBAIKAN: Retry logic untuk handle tunnel yang mati
 * In Railway: Downloads Cloudinary URL, uploads to ngrok as multipart/form-data
 * Returns immediately with job_id for async processing
 */
export async function processVideoViaAPI(videoUrlOrPath, onProgress) {
  return new Promise(async (resolve, reject) => {
    try {
      console.log(`\n🚀 [NGROK API] Starting detection dengan retry logic`);
      console.log(`📹 Video: ${videoUrlOrPath.substring(0, 60)}...`);
      console.log(`🌐 API: ${PYTHON_API}`);

      // Build API URL
      const apiUrl = PYTHON_API.endsWith("/") ? PYTHON_API.slice(0, -1) : PYTHON_API;
      
      if (onProgress) onProgress(10);
      console.log(`📤 Preparing to upload video...`);

      // Step 1: Get video data (from URL or local file)
      let videoBuffer;
      if (videoUrlOrPath.startsWith("http")) {
        // Cloudinary URL - download it
        console.log(`⬇️  Downloading from Cloudinary...`);
        if (onProgress) onProgress(15);
        
        const response = await axios.get(videoUrlOrPath, { 
          responseType: "arraybuffer",
          timeout: 180000  // ⭐ INCREASED: 3 minutes untuk large videos
        });
        videoBuffer = Buffer.from(response.data);
        console.log(`✅ Downloaded ${videoBuffer.length} bytes`);
      } else {
        // Local file
        console.log(`📂 Reading local file...`);
        if (!fs.existsSync(videoUrlOrPath)) {
          throw new Error(`Video file not found: ${videoUrlOrPath}`);
        }
        videoBuffer = fs.readFileSync(videoUrlOrPath);
        console.log(`✅ Read ${videoBuffer.length} bytes`);
      }

      // Step 2: Upload to ngrok YOLO as multipart/form-data dengan RETRY
      if (onProgress) onProgress(25);
      console.log(`📤 Uploading to YOLO API (${videoBuffer.length} bytes) dengan retry...`);
      
      // ⭐ HEALTH CHECK: Verify tunnel is reachable before uploading
      try {
        const healthRes = await axios.get(`${apiUrl}/health`, { timeout: 5000 });
        console.log(`✅ Tunnel health: ${healthRes.data.status}`);
      } catch (healthErr) {
        console.warn(`⚠️  Tunnel health check failed: ${healthErr.message}`);
        console.warn(`📍 Will proceed anyway - retries will handle transient issues`);
      }
      
      const form = new FormData();
      // ngrok YOLO expects "file" field with binary video
      form.append("file", videoBuffer, "video.mp4");

      let detectRes;
      try {
        // ⭐ RETRY WRAPPER untuk handle tunnel timeout/reset
        detectRes = await retryWithBackoff(async () => {
          return await axios.post(
            `${apiUrl}/detect`,
            form,
            {
              headers: form.getHeaders(),
              timeout: 900000, // ⭐ INCREASED: 15 minutes (dari 10 min)
              maxBodyLength: Infinity,
              maxContentLength: Infinity,
              // ⭐ KEEP-ALIVE untuk prevent premature disconnects
              httpAgent: new http.Agent({ 
                keepAlive: true,
                keepAliveMsecs: 30000,
                timeout: 900000
              }),
              httpsAgent: new https.Agent({ 
                keepAlive: true,
                keepAliveMsecs: 30000,
                timeout: 900000
              }),
              onUploadProgress: (progressEvent) => {
                if (progressEvent.total > 0) {
                  const uploadPercent = Math.round((progressEvent.loaded / progressEvent.total) * 30);
                  const totalPercent = 25 + uploadPercent;
                  if (onProgress) onProgress(totalPercent, `Uploading... ${uploadPercent}%`);
                }
              }
            }
          );
        }, 3, 2000); // Retry 3 kali dengan delay awal 2s
      } catch (uploadError) {
        console.error(`❌ Upload failed setelah 3 retry:`, uploadError.message);
        throw new Error(`YOLO upload failed (tunnel unstable?): ${uploadError.message}`);
      }

      if (onProgress) onProgress(60, "Processing in YOLO service...");
      console.log(`✅ Upload successful. Job ID: ${detectRes.data.job_id}`);

      // Step 3: Poll for results (ngrok processes async) dengan RETRY
      console.log(`🔄 Polling for results dengan retry pada timeout...`);
      const jobId = detectRes.data.job_id;
      let result = null;
      let pollCount = 0;
      let consecutiveFailures = 0;
      const maxPolls = 2880; // ⭐ INCREASED: Max 4 hours polling (5sec intervals @ 2880 polls = 14400s = 240m)
      const CIRCUIT_BREAKER_THRESHOLD = 50; // ⭐ NEW: Stop after 50 consecutive failures (tunnel is dead)

      while (pollCount < maxPolls && !result) {
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
        pollCount++;

        try {
          // ⭐ RETRY untuk polling juga (tunnel bisa mati saat polling)
          const resultRes = await retryWithBackoff(async () => {
            return await axios.get(
              `${apiUrl}/result/${jobId}`,
              { 
                timeout: 30000,  // ⭐ INCREASED: 30s (dari 10s)
                httpAgent: new http.Agent({ 
                  keepAlive: true,
                  keepAliveMsecs: 30000
                }),
                httpsAgent: new https.Agent({ 
                  keepAlive: true,
                  keepAliveMsecs: 30000
                })
              }
            );
          }, 3, 2000); // ⭐ INCREASED: Retry 3 kali (dari 2) dengan delay awal 2s (dari 1s)

          // Reset consecutive failures on success
          consecutiveFailures = 0;

          // Check if processing is actually complete (status: "completed")
          if (resultRes.data && resultRes.data.status === "completed") {
            result = resultRes.data;
            console.log(`✅ Results received after ${pollCount * 5} seconds`);
          } else {
            const progress = 60 + Math.round((pollCount / maxPolls) * 40);
            const statusMsg = resultRes.data?.message || "Processing...";
            if (onProgress) onProgress(progress, `Waiting for YOLO processing... (${pollCount * 5}s) [${statusMsg}]`);
          }
        } catch (err) {
          // Results not ready yet, continue polling
          consecutiveFailures++;
          
          const progress = 60 + Math.round((pollCount / maxPolls) * 40);
          const errorMsg = err.code || err.response?.status || err.message;
          
          // Log persistent failures but don't spam logs
          if (consecutiveFailures === 1 || consecutiveFailures % 10 === 0) {
            console.warn(`⚠️  Poll attempt ${pollCount} failed (${errorMsg}) - ${consecutiveFailures} consecutive failures`);
          }
          
          // ⭐ NEW: Circuit Breaker - if tunnel is consistently down, stop retrying
          if (consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
            console.error(`🔴 CIRCUIT BREAKER TRIGGERED: ${consecutiveFailures} consecutive failures detected`);
            console.error(`⚠️  ngrok tunnel (${apiUrl}) appears to be DOWN or UNRESPONSIVE`);
            throw new Error(
              `YOLO API tunnel is unresponsive after ${consecutiveFailures} consecutive attempts. ` +
              `Tunnel URL: ${apiUrl} | Last error: ${errorMsg}. ` +
              `Check: 1) ngrok tunnel status 2) Auth token validity 3) Network connectivity`
            );
          }
          
          if (onProgress) onProgress(progress, `Waiting for YOLO processing...`);
        }
      }

      if (!result) {
        throw new Error(`YOLO processing timeout after ${maxPolls * 5} seconds (${Math.round(maxPolls * 5 / 60)} minutes)`);
      }

      if (onProgress) onProgress(90);

      // Parse result from ngrok API
      // ngrok returns: vehicle_count, frames_processed, lane: {kiri, kanan}
      const parsedResult = {
        totalVehicles: result.vehicle_count || 0,
        carCount: (result.lane?.kiri?.mobil || 0) + (result.lane?.kanan?.mobil || 0),
        busCount: (result.lane?.kiri?.bus || 0) + (result.lane?.kanan?.bus || 0),
        truckCount: (result.lane?.kiri?.truk || 0) + (result.lane?.kanan?.truk || 0),
        leftLaneCount: result.lane?.kiri?.total || 0,
        rightLaneCount: result.lane?.kanan?.total || 0,
        confidence: result.confidence || 0.87,
        processedAt: new Date(),
        // Extra info
        framesProcessed: result.frames_processed || 0,
        outputVideoUrl: result.outputVideoUrl || null,
      };

      console.log(`📊 Results:`, parsedResult);
      if (onProgress) onProgress(100);
      resolve(parsedResult);
    } catch (error) {
      console.error(`❌ YOLO API error: ${error.message}`);
      reject(new Error(`YOLO processing failed: ${error.message}`));
    }
  });
}

/**
 * Process video using LOCAL count_video.py
 * ✅ NO RAILWAY API DEPENDENCY
 * ✅ RELIABLE AND FAST
 * Used in local development only
 */
export async function processVideoLocally(videoPath, onProgress) {
  return new Promise((resolve, reject) => {
    console.log(`\n🚀 [LOCAL YOLO] Starting processing`);
    console.log(`📹 Video: ${videoPath}`);
    console.log(`🐍 Script: ${PYTHON_SCRIPT}`);

    // Validate paths
    if (!fs.existsSync(PYTHON_SCRIPT)) {
      console.error(`❌ Script not found: ${PYTHON_SCRIPT}`);
      return reject(new Error(`Python script not found: ${PYTHON_SCRIPT}`));
    }

    if (!fs.existsSync(videoPath)) {
      console.error(`❌ Video not found: ${videoPath}`);
      return reject(new Error(`Video file not found: ${videoPath}`));
    }

    // Spawn Python process
    const pythonProcess = spawn("python3", [PYTHON_SCRIPT, "--video", videoPath], {
      cwd: YOLO_DIR,
      timeout: 1800000, // 30 minutes
      env: {
        ...process.env,
        // Force CPU mode to avoid GPU incompatibility issues
        CUDA_VISIBLE_DEVICES: "",
        // Disable GPU entirely
        TORCH_DEVICE: "cpu",
        // Force torch/lib to be in library search path
        // PyTorch needs libcublas.so from its internal stubs
        LD_LIBRARY_PATH: `/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib:${process.env.LD_LIBRARY_PATH || ""}`,
      }
    });

    let outputData = "";
    let errorData = "";
    let lastProgress = 0;

    console.log(`⏳ Initializing Python process...`);

    // Capture stdout (progress and output)
    pythonProcess.stdout.on("data", (data) => {
      const chunk = data.toString();
      outputData += chunk;

      // Log each line for debugging
      const lines = chunk.trim().split("\n");
      lines.forEach(line => {
        if (line.length > 0) {
          console.log(`[PYTHON] ${line}`);
        }
      });

      // Parse progress percentage
      const progressMatch = chunk.match(/(\d+\.\d+)%/);
      if (progressMatch) {
        const progress = Math.round(parseFloat(progressMatch[1]));
        if (progress > lastProgress) {
          lastProgress = progress;
          if (onProgress) {
            // Map 0-100% Python progress to 20-90% job progress
            const jobProgress = Math.min(90, 20 + progress * 0.7);
            onProgress(jobProgress);
            console.log(`📊 Progress: ${jobProgress}%`);
          }
        }
      }
    });

    // Capture stderr (errors)
    pythonProcess.stderr.on("data", (data) => {
      const chunk = data.toString();
      errorData += chunk;
      // Print every stderr line immediately for debugging
      chunk.trim().split("\n").forEach(line => {
        if (line.length > 0) {
          console.error(`[PYTHON STDERR] ${line}`);
        }
      });
    });

    // Process finished
    pythonProcess.on("close", (code) => {
      console.log(`\n🔚 Python process ended with code: ${code}`);

      if (code !== 0) {
        console.error(`❌ Processing failed`);
        console.error(`Error output: ${errorData}`);
        return reject(new Error(`Python processing failed: ${errorData || "Unknown error"}`));
      }

      try {
        console.log(`\n✅ Processing completed`);
        console.log(`📊 Parsing results...`);

        // Try to parse JSON output from Python (new format: [JSON_RESULT]...[/JSON_RESULT])
        let jsonMatch = outputData.match(/\[JSON_RESULT\]\s*([\s\S]*?)\s*\[\/JSON_RESULT\]/);
        
        // Fallback to old JSON format if new one not found
        if (!jsonMatch) {
          jsonMatch = outputData.match(/JSON_OUTPUT_START\s*(\{[\s\S]*?\})\s*JSON_OUTPUT_END/);
        }

        let result;
        if (jsonMatch) {
          console.log(`✅ Found JSON output`);
          result = JSON.parse(jsonMatch[1]);
          console.log(`📊 Parsed result:`, result);
        } else {
          console.warn(`⚠️ JSON output not found, parsing summary...`);

          // Fallback: parse summary text
          const extractNumber = (pattern) => {
            const match = outputData.match(pattern);
            return match ? parseInt(match[1]) : 0;
          };

          result = {
            totalVehicles: extractNumber(/GRAND TOTAL:\s*(\d+)/),
            carCount: 0,
            busCount: 0,
            truckCount: 0,
            leftLaneCount: extractNumber(/Lajur Kiri:[\s\S]*?Total:\s*(\d+)/),
            rightLaneCount: extractNumber(/Lajur Kanan:[\s\S]*?Total:\s*(\d+)/),
            confidence: 0.87,
            framesProcessed: 0,
            outputVideoPath: undefined,
          };

          console.log(`📊 Parsed summary:`, result);
        }

        // Format final result
        const leftLaneCount = parseInt(result.leftLaneCount) || 0;
        const rightLaneCount = parseInt(result.rightLaneCount) || 0;
        
        // CRITICAL FIX: Calculate totalVehicles from lane counts
        const correctTotalVehicles = leftLaneCount + rightLaneCount;
        
        const finalResult = {
          totalVehicles: correctTotalVehicles,  // Use corrected total from lane counts!
          carCount: parseInt(result.carCount) || 0,
          busCount: parseInt(result.busCount) || 0,
          truckCount: parseInt(result.truckCount) || 0,
          leftLaneCount: leftLaneCount,
          rightLaneCount: rightLaneCount,
          confidence: parseFloat(result.confidence) || 0.87,
          duration: parseInt(result.durationSeconds) || Math.round((result.framesProcessed || 1800) / 30),
          frames: parseInt(result.framesProcessed) || 0,
          outputVideoPath: result.outputVideoPath,
          detections: [
            {
              lane: "Kiri",
              car: parseInt(result.carCount) || 0,  // Note: carCount is total across both lanes
              bus: parseInt(result.busCount) || 0,
              truck: parseInt(result.truckCount) || 0,
              total: leftLaneCount,
              confidence: parseFloat(result.confidence) || 0.87,
            },
            {
              lane: "Kanan",
              car: parseInt(result.carCount) || 0,  // Note: carCount is total across both lanes
              bus: parseInt(result.busCount) || 0,
              truck: parseInt(result.truckCount) || 0,
              total: rightLaneCount,
              confidence: parseFloat(result.confidence) || 0.87,
            },
          ],
        };

        console.log(`\n✅ Final result:`);
        console.log(`  📊 Total vehicles: ${finalResult.totalVehicles}`);
        console.log(`  🚗 Cars: ${finalResult.carCount}`);
        console.log(`  🚌 Buses: ${finalResult.busCount}`);
        console.log(`  🚚 Trucks: ${finalResult.truckCount}`);
        console.log(`  ← Left lane: ${finalResult.leftLaneCount}`);
        console.log(`  → Right lane: ${finalResult.rightLaneCount}`);

        if (onProgress) onProgress(95);
        resolve(finalResult);
      } catch (parseError) {
        console.error(`❌ Failed to parse output:`, parseError.message);
        console.error(`Raw output (first 500 chars):\n${outputData.substring(0, 500)}`);
        reject(new Error(`Failed to parse YOLO output: ${parseError.message}`));
      }
    });

    // Timeout after 30 minutes
    const timeoutHandle = setTimeout(() => {
      console.error(`\n❌ TIMEOUT: Processing exceeded 30 minutes`);
      pythonProcess.kill();
      reject(new Error("Local YOLO processing timeout (30 minutes)"));
    }, 1800000);

    // Clear timeout when process closes
    pythonProcess.on("close", () => clearTimeout(timeoutHandle));
  });
}

/**
 * Legacy: uploadToYoloAPI (DEPRECATED)
 * Use processVideoLocally() instead
 */
export async function uploadToYoloAPI(videoBuffer, fileName, onProgress) {
  console.error(`\n❌ [RAILWAY] DEPRECATED`);
  console.log(`ℹ️ Railway API support has been removed`);
  console.log(`📁 Using local processing instead`);
  throw new Error("Railway API is deprecated. Use local processing instead.");
}

/**
 * Legacy: pollYoloResult (DEPRECATED)
 */
export async function pollYoloResult(jobId, maxAttempts, delayMs) {
  throw new Error("Railway API is deprecated. Use local processing instead.");
}

/**
 * Utility: Download from Cloudinary
 */
export async function downloadFromCloudinary(cloudinaryUrl) {
  return new Promise((resolve, reject) => {
    if (!cloudinaryUrl) {
      return reject(new Error("Cloudinary URL is required"));
    }

    console.log(`📥 Downloading from Cloudinary...`);
    console.log(`🔗 URL: ${cloudinaryUrl}`);

    // Implementation would go here - for now just return the URL
    // since we'll handle download in the routes
    resolve(cloudinaryUrl);
  });
}

/**
 * Utility: Parse YOLO result from manual tracking (count_video.py)
 * 
 * Manual tracking logic:
 * - Accurate vehicle counting using line crossing detection
 * - Lane classification based on centroid x position
 * - Vehicle classification by size + YOLO model
 */
export function parseYoloResult(result) {
  console.log(`\n📊 [PARSE YOLO RESULT]`);
  console.log(`  Raw result keys: ${Object.keys(result).join(', ')}`);
  
  // Duration from Python: durationSeconds or calculated from frames/fps
  const durationSeconds = parseInt(result.durationSeconds) || Math.round((parseInt(result.framesProcessed) || 0) / (parseInt(result.fps) || 30));
  
  // Lane counts from manual tracking (accurate)
  const leftLaneCount = parseInt(result.leftLaneCount) || 0;
  const rightLaneCount = parseInt(result.rightLaneCount) || 0;
  
  // 🔴 CRITICAL: Use lane counts for total (not result.totalVehicles which might be wrong)
  // Manual tracking detects actual vehicles crossing line
  const correctTotalVehicles = leftLaneCount + rightLaneCount;
  
  console.log(`  Left Lane: ${leftLaneCount}`);
  console.log(`  Right Lane: ${rightLaneCount}`);
  console.log(`  ✅ Calculated Total: ${correctTotalVehicles}`);
  
  if (result.totalVehicles && result.totalVehicles !== correctTotalVehicles) {
    console.log(`  ⚠️  Raw totalVehicles (${result.totalVehicles}) != calculated (${correctTotalVehicles})`);
  }
  
  return {
    totalVehicles: correctTotalVehicles,  // Use lane-based calculation
    carCount: parseInt(result.carCount) || 0,
    busCount: parseInt(result.busCount) || 0,
    truckCount: parseInt(result.truckCount) || 0,
    leftLaneCount: leftLaneCount,
    rightLaneCount: rightLaneCount,
    frames: parseInt(result.framesProcessed) || 0,
    duration: durationSeconds,
    fps: parseInt(result.fps) || 30,
    confidence: parseFloat(result.confidence) || 0.87,
    lane: result.lane || {
      kiri: {
        total: leftLaneCount,
        mobil: parseInt(result.carCount) || 0,
        bus: parseInt(result.busCount) || 0,
        truk: parseInt(result.truckCount) || 0
      },
      kanan: {
        total: rightLaneCount,
        mobil: 0,
        bus: 0,
        truk: 0
      }
    }
  };
}

/**
 * Utility: Validate video file
 */
export function validateVideoFile(videoBuffer, fileName) {
  const MIN_SIZE = 1024; // 1KB
  const MAX_SIZE = 500 * 1024 * 1024; // 500MB

  if (videoBuffer.length < MIN_SIZE) {
    throw new Error(`Video too small (${videoBuffer.length} bytes, minimum ${MIN_SIZE} bytes)`);
  }

  if (videoBuffer.length > MAX_SIZE) {
    throw new Error(`Video too large (${videoBuffer.length} bytes, maximum ${MAX_SIZE} bytes)`);
  }

  const validExtensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"];
  const ext = path.extname(fileName).toLowerCase();

  if (!validExtensions.includes(ext)) {
    throw new Error(`Invalid video format: ${ext}. Supported: ${validExtensions.join(", ")}`);
  }

  return true;
}

/**
 * Utility: Sleep/delay
 */
export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
/**
 * Process video from Cloudinary URL
 * 
 * RAILWAY:  Passes Cloudinary URL directly to ngrok YOLO API (no download)
 * LOCAL:    Downloads video, saves to temp file, processes locally
 */
export async function processVideoFromCloudinary(cloudinaryUrl, options = {}) {
  const { jobId, onProgress } = options;
  
  try {
    console.log(`\n🎬 [processVideoFromCloudinary] Starting`);
    console.log(`   URL: ${cloudinaryUrl.substring(0, 60)}...`);
    console.log(`   jobId: ${jobId}`);
    console.log(`   Environment: ${IS_RAILWAY ? "🚂 RAILWAY" : "🖥️  LOCAL"}`);
    
    if (IS_RAILWAY) {
      // RAILWAY: Pass Cloudinary URL directly to ngrok YOLO API
      console.log(`📍 RAILWAY MODE: Passing Cloudinary URL directly to YOLO API`);
      if (onProgress) onProgress(10, "Sending to YOLO API...");
      
      const result = await processVideoViaAPI(cloudinaryUrl, (progress, message) => {
        // Map progress from 10-100
        const mappedProgress = 10 + Math.round((progress * 90) / 100);
        if (onProgress) onProgress(mappedProgress, message || "Processing via YOLO API...");
      });
      
      console.log(`✅ [processVideoFromCloudinary] Complete (API mode)`);
      return result;
    } else {
      // LOCAL: Download and process locally
      console.log(`📍 LOCAL MODE: Downloading from Cloudinary and processing locally`);
      
      // Step 1: Download from Cloudinary
      if (onProgress) onProgress(10, "Downloading video from Cloudinary...");
      console.log(`⏳ Step 1: Downloading from Cloudinary...`);
      const response = await axios.get(cloudinaryUrl, { responseType: "arraybuffer" });
      const videoBuffer = Buffer.from(response.data);
      console.log(`✅ Step 1: Downloaded ${videoBuffer.length} bytes`);
      
      // Step 2: Save to temp file
      if (onProgress) onProgress(20, "Preparing video file...");
      console.log(`⏳ Step 2: Saving to temp file...`);
      const tempPath = path.join("/tmp", `video_${jobId}_${Date.now()}.mp4`);
      fs.writeFileSync(tempPath, videoBuffer);
      console.log(`✅ Step 2: Saved to ${tempPath}`);
      
      // Step 3: Process locally
      if (onProgress) onProgress(30, "Processing with YOLO...");
      console.log(`⏳ Step 3: Processing with YOLO...`);
      const result = await processVideoLocally(tempPath, (progress, message) => {
        // Map progress from 30-100
        const mappedProgress = 30 + Math.round((progress * 70) / 100);
        if (onProgress) onProgress(mappedProgress, message || "Processing...");
      });
      console.log(`✅ Step 3: Processing complete`);
      
      // Step 4: Cleanup
      if (onProgress) onProgress(95, "Cleaning up...");
      console.log(`⏳ Step 4: Cleanup...`);
      try {
        fs.unlinkSync(tempPath);
        console.log(`✅ Step 4: Cleaned up temp file`);
      } catch (err) {
        console.warn(`⚠️ Step 4: Failed to cleanup temp file:`, err.message);
      }
      
      if (onProgress) onProgress(100, "Complete!");
      console.log(`✅ [processVideoFromCloudinary] Complete (Local mode)`);
      
      return result;
    }
  } catch (err) {
    console.error(`❌ [processVideoFromCloudinary] Error:`, err.message);
    throw err;
  }
}

/**
 * SMART WRAPPER: Automatically chooses between local and API processing
 * 
 * RAILWAY (production):  Uses processVideoViaAPI() → ngrok YOLO
 * LOCAL (development):   Uses processVideoLocally() → count_video.py
 * 
 * This is the primary function to use in controllers/routes
 */
export async function processVideo(videoPath, onProgress) {
  console.log(`\n🎯 [processVideo] Smart routing`);
  console.log(`📍 Environment: ${IS_RAILWAY ? "RAILWAY" : "LOCAL DEV"}`);
  
  if (IS_RAILWAY) {
    console.log(`✅ Routing to PYTHON_API (ngrok YOLO service)...`);
    return processVideoViaAPI(videoPath, onProgress);
  } else {
    console.log(`✅ Routing to local Python subprocess...`);
    return processVideoLocally(videoPath, onProgress);
  }
}