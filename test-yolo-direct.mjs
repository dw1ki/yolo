import axios from 'axios';
import fs from 'fs';
import FormData from 'form-data';

async function test() {
  try {
    const testVideoPath = '/mnt/data2/pktj/backend/yolo/test_small.mp4';
    
    if (!fs.existsSync(testVideoPath)) {
      console.log('❌ Test video not found');
      return;
    }
    
    const videoSize = fs.statSync(testVideoPath).size;
    console.log(`📹 Testing with: test_small.mp4`);
    console.log(`📊 File size: ${(videoSize / 1024 / 1024).toFixed(2)} MB`);
    
    const form = new FormData();
    form.append('file', fs.createReadStream(testVideoPath));
    
    console.log('🚀 Uploading to YOLO API at http://localhost:8000/detect...');
    const detectRes = await axios.post('http://localhost:8000/detect', form, {
      headers: form.getHeaders(),
      timeout: 30000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity
    });
    
    console.log('✅ Upload successful');
    console.log('Response:', JSON.stringify(detectRes.data, null, 2));
    
    const jobId = detectRes.data.job_id;
    console.log(`\n⏳ Job started (ID: ${jobId}), polling for progress...`);
    
    // Poll for results (15 times, 5 seconds apart = 75 seconds total)
    for (let i = 0; i < 15; i++) {
      await new Promise(r => setTimeout(r, 5000));
      
      try {
        const resultRes = await axios.get(`http://localhost:8000/result/${jobId}`, { timeout: 10000 });
        const job = resultRes.data;
        console.log(`[Poll ${i+1}] Status: ${job.status} | Progress: ${job.progress}%`);
        
        if (job.status === 'completed') {
          console.log('\n✅ Job completed!');
          console.log('Vehicle count:', job.vehicle_count);
          console.log('Frames processed:', job.frames_processed);
          console.log('Lane breakdown:', JSON.stringify(job.lane, null, 2));
          return;
        }
      } catch (e) {
        console.log(`[Poll ${i+1}] Error: ${e.message}`);
      }
    }
    
    console.log('⏱️  Polling timeout after 75 seconds');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

test();
