import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';

const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY5NWRkMjQ2MDQwMjM0NDk4N2E0ZWM3NSIsInJvbGUiOiJhZG1pbiIsImlhdCI6MTc2Nzc3ODIwNiwiZXhwIjoxNzY3ODY0NjA2fQ.T5F2wUMw8sDveELAolnd98cQ7mL_TBTXn_5XbwTah6w";

async function testUpload(videoPath, testName) {
  try {
    console.log(`\n${testName}`);
    console.log("=".repeat(50));
    
    if (!fs.existsSync(videoPath)) {
      console.log(`❌ Video not found: ${videoPath}`);
      return;
    }
    
    const stats = fs.statSync(videoPath);
    console.log(`Testing with: ${path.basename(videoPath)} (${stats.size / 1024 / 1024}MB)`);
    
    const form = new FormData();
    form.append('video', fs.createReadStream(videoPath));

    console.log("Sending request...");
    const response = await axios.post(
      'http://localhost:5000/api/detect/yolo/start-job',
      form,
      {
        headers: {
          ...form.getHeaders(),
          'Authorization': `Bearer ${token}`
        },
        timeout: 60000,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
          if (percent % 20 === 0) {
            console.log(`Upload progress: ${percent}%`);
          }
        }
      }
    );

    console.log("✅ Success!");
    console.log("Response:", JSON.stringify(response.data, null, 2));
  } catch (err) {
    console.error("❌ Error:");
    if (err.response) {
      console.error("Status:", err.response.status);
      console.error("Data:", JSON.stringify(err.response.data, null, 2));
    } else {
      console.error("Message:", err.message);
      console.error("Code:", err.code);
    }
  }
}

console.log("\n🧪 YOLO Upload Endpoint Tests\n");

// Test 1: Invalid video (random binary)
await testUpload('/tmp/test-video.mp4', 'Test 1: Invalid video (random binary)');

// Test 2: Valid video (if exists)
await testUpload('/tmp/valid-video.mp4', 'Test 2: Valid video (27MB)');

console.log("\n" + "=".repeat(50));
console.log("✅ All tests completed");

