/**
 * Test ngrok YOLO API with multipart/form-data
 */

import axios from 'axios';
import FormData from 'form-data';

const NGROK_API = 'https://unobscurely-blossomy-ameer.ngrok-free.dev/';
const CLOUDINARY_URL = 'https://res.cloudinary.com/dgiv24lgt/video/upload/v1768560608/siang_online-video-cutter.com_of3nny.mp4';

async function testNgrokMultipart() {
  console.log('\n🧪 Testing ngrok YOLO API with multipart/form-data\n');
  console.log('NGROK API:', NGROK_API);
  console.log('Cloudinary URL:', CLOUDINARY_URL.substring(0, 60) + '...\n');

  try {
    // Step 1: Download video from Cloudinary
    console.log('📍 Step 1: Downloading from Cloudinary');
    const downloadRes = await axios.get(CLOUDINARY_URL, { 
      responseType: 'arraybuffer',
      timeout: 30000 
    });
    const videoBuffer = Buffer.from(downloadRes.data);
    console.log(`✅ Downloaded ${videoBuffer.length} bytes`);

    // Step 2: Upload to ngrok as multipart/form-data
    console.log('\n📍 Step 2: Uploading to ngrok YOLO API');
    const form = new FormData();
    form.append('file', videoBuffer, 'video.mp4');

    const detectRes = await axios.post(
      `${NGROK_API}detect`,
      form,
      {
        headers: form.getHeaders(),
        timeout: 30000, // shorter timeout for test
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
      }
    );

    console.log('✅ Detection successful!');
    console.log('Response:', JSON.stringify(detectRes.data, null, 2));

    console.log('\n✅ Test passed! Railway can successfully call ngrok YOLO API');

  } catch (err) {
    console.error('\n❌ Error:', err.message);
    if (err.response) {
      console.error('Status:', err.response.status);
      console.error('Data:', err.response.data);
    }
  }
}

testNgrokMultipart();
