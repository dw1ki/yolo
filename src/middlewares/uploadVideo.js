import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// folder sementara untuk upload
const upload = multer({ dest: "tmp/" });

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

export default upload;
