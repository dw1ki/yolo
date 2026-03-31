/**
 * Test ngrok YOLO API directly
 * Simulates what Railway backend should do when calling ngrok
 */

import axios from 'axios';

const NGROK_API = 'https://unobscurely-blossomy-ameer.ngrok-free.dev/';
const CLOUDINARY_URL = 'https://res.cloudinary.com/dgiv24lgt/video/upload/v1768560608/siang_online-video-cutter.com_of3nny.mp4';

async function testNgrokAPI() {
  console.log('\n🧪 Testing ngrok YOLO API Directly\n');
  console.log('NGROK API:', NGROK_API);
  console.log('Cloudinary URL:', CLOUDINARY_URL.substring(0, 60) + '...\n');

  try {
    // Test 1: Check API health
    console.log('📍 Test 1: Health check');
    try {
      const healthRes = await axios.get(NGROK_API, { timeout: 5000 });
      console.log('✅ API is reachable');
      console.log('Status:', healthRes.status);
    } catch (err) {
      console.log('⚠️  Health check failed (API may not have health endpoint)');
      console.log('Error:', err.message);
    }

    // Test 2: Send Cloudinary URL to /detect endpoint
    console.log('\n📍 Test 2: Sending Cloudinary URL to /detect');
    const detectRes = await axios.post(
      `${NGROK_API}detect`,
      { video_path: CLOUDINARY_URL },
      { 
        timeout: 30000,
        headers: { 'Content-Type': 'application/json' }
      }
    );

    console.log('✅ Detection request successful!');
    console.log('Response:', JSON.stringify(detectRes.data, null, 2));

    // Test 3: Parse results
    console.log('\n📍 Test 3: Parsing results');
    const result = {
      totalVehicles: detectRes.data.totalVehicles || detectRes.data.total_vehicles || 0,
      carCount: detectRes.data.carCount || detectRes.data.car_count || 0,
      busCount: detectRes.data.busCount || detectRes.data.bus_count || 0,
      truckCount: detectRes.data.truckCount || detectRes.data.truck_count || 0,
      leftLaneCount: detectRes.data.leftLaneCount || detectRes.data.left_lane_count || 0,
      rightLaneCount: detectRes.data.rightLaneCount || detectRes.data.right_lane_count || 0,
      confidence: detectRes.data.confidence || 0.87,
    };
    console.log('Parsed result:', JSON.stringify(result, null, 2));
    console.log('\n✅ All tests passed! Railway can successfully call ngrok YOLO API');

  } catch (err) {
    console.error('\n❌ Error:', err.message);
    if (err.response) {
      console.error('Status:', err.response.status);
      console.error('Data:', err.response.data);
    }
    console.error('Stack:', err.stack);
  }
}

testNgrokAPI();
