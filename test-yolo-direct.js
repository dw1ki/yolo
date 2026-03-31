const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

async function test() {
  try {
    // Create a small test video (just use an existing one if available)
    const testVideoPath = '/mnt/data2/pktj/backend/yolo/uploads/siang (online-video-cutter.com).mp4';
    
    if (!fs.existsSync(testVideoPath)) {
      console.log('❌ Test video not found at ' + testVideoPath);
      // Try another path
      const existingUploads = fs.readdirSync('/mnt/data2/pktj/backend/yolo/uploads/').filter(f => f.endsWith('.mp4'));
      if (existingUploads.length > 0) {
        console.log(`📹 Found video: ${existingUploads[0]}`);
        testVideoPath = `/mnt/data2/pktj/backend/yolo/uploads/${existingUploads[0]}`;
      } else {
        console.log('❌ No test videos found');
        return;
      }
    }
    
    console.log(`📹 Testing with video: ${testVideoPath}`);
    console.log(`📊 File size: ${(fs.statSync(testVideoPath).size / 1024 / 1024).toFixed(2)} MB`);
    
    // Upload to ngrok (not local API - ngrok tunnel)
    const form = new FormData();
    form.append('file', fs.createReadStream(testVideoPath));
    
    console.log('🚀 Uploading to YOLO API at http://localhost:8000...');
    const detectRes = await axios.post('http://localhost:8000/detect', form, {
      headers: form.getHeaders(),
      timeout: 30000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity
    });
    
    console.log('✅ Upload successful');
    console.log('Response:', JSON.stringify(detectRes.data, null, 2));
    
    const jobId = detectRes.data.job_id;
    console.log(`\n⏳ Polling for results (job_id: ${jobId})...`);
    
    // Poll for results
    let attempts = 0;
    while (attempts < 5) {
      await new Promise(r => setTimeout(r, 3000));
      attempts++;
      
      try {
        const resultRes = await axios.get(`http://localhost:8000/result/${jobId}`, { timeout: 10000 });
        const job = resultRes.data;
        console.log(`[Poll ${attempts}] Status: ${job.status} | Progress: ${job.progress}%`);
        
        if (job.status === 'completed') {
          console.log('✅ Job completed!');
          console.log('Vehicle count:', job.vehicle_count);
          console.log('Frames processed:', job.frames_processed);
          console.log('Lane breakdown:', job.lane);
          break;
        }
      } catch (e) {
        console.log(`[Poll ${attempts}] Error: ${e.message}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response:', error.response.data);
    }
  }
}

test();
