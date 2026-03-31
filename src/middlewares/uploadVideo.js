import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ⭐ Use memory storage for Vercel (read-only filesystem)
let upload;

try {
  // Production (Vercel): memory storage
  if (process.env.NODE_ENV === "production") {
    upload = multer({ 
      storage: multer.memoryStorage(),
      limits: { fileSize: 500 * 1024 * 1024 } // 500MB
    });
    console.log("[MULTER] Using memory storage for production");
  } else {
    // Local development: disk storage
    const storage = multer.diskStorage({
      destination: (req, file, cb) => {
        const tmpFolder = path.join(__dirname, "../../tmp");
        if (!fs.existsSync(tmpFolder)) {
          fs.mkdirSync(tmpFolder, { recursive: true });
        }
        cb(null, tmpFolder);
      },
      filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
      }
    });
    
    upload = multer({ 
      storage: storage,
      limits: { fileSize: 500 * 1024 * 1024 }
    });
    console.log("[MULTER] Using disk storage for local development");
  }
} catch (err) {
  console.error("[MULTER] Error initializing multer:", err.message);
  // Fallback to memory storage
  upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 500 * 1024 * 1024 }
  });
  console.log("[MULTER] Fallback to memory storage due to error");
}

// ⭐ NEW: Save to local storage instead of Cloudinary
export const saveToLocalStorage = async (filePath, originalName) => {
  try {
    // Create path: backend/yolo/input_videos
    const inputFolder = path.join(__dirname, "../../yolo/input_videos");
    
    if (!fs.existsSync(inputFolder)) {
      fs.mkdirSync(inputFolder, { recursive: true });
      console.log(`✅ Created input folder: ${inputFolder}`);
    }

    // Generate unique filename with timestamp
    const timestamp = Date.now();
    const safeFileName = `video_${timestamp}_${originalName.replace(/[^\w.-]/g, '_')}`;
    const destinationPath = path.join(inputFolder, safeFileName);

    // Move file from tmp to backend/yolo/input_videos
    fs.copyFileSync(filePath, destinationPath);
    fs.unlinkSync(filePath); // Remove temp file

    console.log(`✅ Video saved locally: ${destinationPath}`);

    return {
      filePath: destinationPath,
      fileName: safeFileName,
      relativePath: `yolo/input_videos/${safeFileName}`,
      fileSize: fs.statSync(destinationPath).size
    };
  } catch (err) {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    throw err;
  }
};

// ⭐ NEW: Save output video from YOLO API to local storage
export const saveOutputVideo = async (outputVideoUrl, detectionId, originalFileName) => {
  try {
    const outputFolder = path.join(__dirname, "../../yolo/output_videos");
    
    if (!fs.existsSync(outputFolder)) {
      fs.mkdirSync(outputFolder, { recursive: true });
      console.log(`✅ Created output folder: ${outputFolder}`);
    }

    // Generate unique filename with timestamp
    const timestamp = Date.now();
    const sanitizedName = originalFileName.replace(/[^\w.-]/g, '_');
    const safeFileName = `output_${timestamp}_${detectionId.substring(0, 8)}_${sanitizedName}`;
    const destinationPath = path.join(outputFolder, safeFileName);

    // If outputVideoUrl is a URL, download it; if it's a local path, copy it
    if (outputVideoUrl.startsWith('http://') || outputVideoUrl.startsWith('https://')) {
      console.log(`📥 [SAVE OUTPUT] Downloading from URL: ${outputVideoUrl}`);
      const response = await axios.get(outputVideoUrl, { responseType: 'stream' });
      
      return new Promise((resolve, reject) => {
        const fileStream = fs.createWriteStream(destinationPath);
        response.data.pipe(fileStream);
        fileStream.on('finish', () => {
          console.log(`✅ [SAVE OUTPUT] Output video saved: ${destinationPath}`);
          resolve({
            filePath: destinationPath,
            fileName: safeFileName,
            relativePath: `yolo/output_videos/${safeFileName}`,
            fileSize: fs.statSync(destinationPath).size
          });
        });
        fileStream.on('error', reject);
      });
    } else {
      // It's a local path - copy it
      console.log(`📋 [SAVE OUTPUT] Copying from local path: ${outputVideoUrl}`);
      
      if (!fs.existsSync(outputVideoUrl)) {
        throw new Error(`Output video not found at: ${outputVideoUrl}`);
      }

      fs.copyFileSync(outputVideoUrl, destinationPath);
      console.log(`✅ [SAVE OUTPUT] Output video copied: ${destinationPath}`);

      return {
        filePath: destinationPath,
        fileName: safeFileName,
        relativePath: `yolo/output_videos/${safeFileName}`,
        fileSize: fs.statSync(destinationPath).size
      };
    }
  } catch (err) {
    console.error(`❌ [SAVE OUTPUT] Error saving output video:`, err.message);
    throw err;
  }
};

export default upload;
