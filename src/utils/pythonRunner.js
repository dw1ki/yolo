/**
 * Python script runner untuk menjalankan YOLO processing
 * Handles: count_video.py, api.py, dll
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// Detect if we're running on Vercel/serverless - MUST BE FIRST
const isServerless = () => {
  // Check multiple indicators
  if (process.env.VERCEL_RUNTIME) return true;
  if (process.env.NOW_REGION) return true;
  if (process.env.LAMBDA_TASK_ROOT) return true;
  if (/var[\/\\]task/.test(process.cwd())) return true;
  if (process.cwd().includes('var/task')) return true;
  return false;
};

// NEVER use path.resolve on Vercel - use simple strings
const YOLO_DIR = isServerless() ? '/var/task/backend/yolo' : './backend/yolo';
const TEMP_DIR = isServerless() ? '/tmp' : './backend/tmp';

// Ensure temp directory exists - but only on local machines
// On Vercel/serverless, /tmp always exists; on local, create if needed
function ensureTempDir() {
  // Never throw errors for mkdir - filesystem might be read-only
  try {
    if (!isServerless() && !fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
      console.log(`✅ Created temp directory: ${TEMP_DIR}`);
    }
  } catch (err) {
    // Silent fail - Vercel filesystem is read-only
    console.log(`ℹ️ Skipping mkdir (read-only FS or already exists): ${err.code}`);
  }
}

/**
 * Run count_video.py dengan video URL dari Cloudinary
 * @param {string} inputUrl - Cloudinary video URL
 * @param {string} outputPath - Output path untuk video hasil
 * @param {function} onProgress - Callback untuk progress updates
 * @returns {Promise} Result dari processing
 */
export async function runCountVideo(inputUrl, outputPath, onProgress = null) {
  return new Promise((resolve, reject) => {
    // Ensure temp directory exists when needed
    ensureTempDir();
    
    console.log(`🎬 [Python] Running count_video.py`);
    console.log(`   Input: ${inputUrl}`);
    console.log(`   Output: ${outputPath}`);

    try {
      // Download video dari Cloudinary ke temp file
      const tempInputPath = path.join(TEMP_DIR, `input_${Date.now()}.mp4`);
      
      // Spawn Python process
      const python = spawn('python3', [
        path.join(YOLO_DIR, 'count_video.py'),
        '--input', inputUrl,
        '--output', outputPath,
        '--temp', tempInputPath,
      ]);

      let stdoutData = '';
      let stderrData = '';

      // Capture stdout untuk progress tracking
      python.stdout.on('data', (data) => {
        const output = data.toString();
        stdoutData += output;
        console.log(`📤 [Python stdout]: ${output.trim()}`);

        // Parse progress dari output
        // Expected format: "PROGRESS: 50" atau "COUNT: {'car': 10, 'truck': 5}"
        if (output.includes('PROGRESS:')) {
          const match = output.match(/PROGRESS:\s*(\d+)/);
          if (match && onProgress) {
            onProgress(parseInt(match[1]));
          }
        }
      });

      // Capture stderr untuk error messages
      python.stderr.on('data', (data) => {
        const output = data.toString();
        stderrData += output;
        console.error(`❌ [Python stderr]: ${output.trim()}`);
      });

      // Handle process completion
      python.on('close', (code) => {
        if (code === 0) {
          console.log(`✅ [Python] count_video.py completed successfully`);
          
          // Parse results dari stdout
          let results = {
            success: true,
            videoUrl: outputPath,
            counts: {},
            statistics: {}
          };

          try {
            // Extract counts dari stdout
            const countMatch = stdoutData.match(/COUNT:\s*({.*?})/);
            if (countMatch) {
              results.counts = JSON.parse(countMatch[1]);
            }
          } catch (err) {
            console.warn('⚠️ Could not parse counts:', err.message);
          }

          // Cleanup temp file
          if (fs.existsSync(tempInputPath)) {
            fs.unlinkSync(tempInputPath);
          }

          resolve(results);
        } else {
          const errorMsg = stderrData || `Python process exited with code ${code}`;
          console.error(`❌ [Python] Error: ${errorMsg}`);
          reject(new Error(`YOLO processing failed: ${errorMsg}`));
        }
      });

      // Handle process error
      python.on('error', (err) => {
        console.error(`❌ [Python] Process error:`, err.message);
        // Special handling for ENOENT (Python not found)
        if (err.code === 'ENOENT') {
          const message = 'Python3 is not installed in this runtime environment. Please configure Railway with Python runtime.';
          console.error(`⚠️ [Python] ${message}`);
          reject(new Error(`PYTHON_NOT_FOUND: ${message}`));
        } else {
          reject(new Error(`Failed to start Python process: ${err.message}`));
        }
      });

      // Timeout setelah 1 jam
      const timeout = setTimeout(() => {
        python.kill();
        reject(new Error('YOLO processing timeout (> 1 hour)'));
      }, 60 * 60 * 1000);

      python.on('close', () => clearTimeout(timeout));

    } catch (err) {
      console.error(`❌ [Python] Unexpected error:`, err.message);
      reject(err);
    }
  });
}

/**
 * Check if Python dan required libraries tersedia
 */
export async function checkPythonSetup() {
  return new Promise((resolve) => {
    const python = spawn('python3', ['-c', 'import cv2, torch; print("OK")']);

    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Python environment ready');
        resolve(true);
      } else {
        console.warn('⚠️ Python setup issue:', error);
        resolve(false);
      }
    });

    // Timeout 10 seconds
    setTimeout(() => {
      python.kill();
      resolve(false);
    }, 10000);
  });
}

export default {
  runCountVideo,
  checkPythonSetup,
};
