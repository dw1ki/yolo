/**
 * Test script untuk job queue system
 * Gunakan: node test-job-queue.js
 */

import axios from 'axios';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const API_URL = 'http://localhost:5000/api';
const JWT_SECRET = 'supersecretpktj';

// Import jwt library untuk generate token
import jwt from 'jsonwebtoken';

// Generate test JWT token
const generateToken = () => {
  return jwt.sign(
    { userId: 'test-user-123', email: 'test@example.com' },
    JWT_SECRET,
    { expiresIn: '1h' }
  );
};

const token = generateToken();
console.log('🔑 Generated test token:', token.substring(0, 30) + '...\n');

// Test functions
async function testJobQueue() {
  console.log('==========================================');
  console.log('🧪 Testing Job Queue System');
  console.log('==========================================\n');

  try {
    // Test 1: Check health of backend
    console.log('1️⃣ Checking backend health...');
    try {
      const healthRes = await axios.get('http://localhost:5000', { timeout: 5000 });
      console.log('✅ Backend responding');
    } catch (err) {
      console.log('⚠️ Backend health check:', err.message);
    }

    // Test 2: Mock endpoint
    console.log('\n2️⃣ Testing mock endpoint...');
    try {
      const mockRes = await axios.post(
        `${API_URL}/detect/yolo/mock`,
        {},
        {
          headers: { Authorization: `Bearer ${token}` },
          timeout: 10000
        }
      );
      console.log('✅ Mock endpoint working');
      console.log('   Response:', mockRes.data);
    } catch (err) {
      console.log('❌ Mock endpoint failed:', err.message);
      if (err.response) {
        console.log('   Status:', err.response.status);
        console.log('   Data:', err.response.data);
      }
    }

    // Test 3: Create job via mock endpoint
    console.log('\n3️⃣ Creating job via mock endpoint...');
    let jobId = null;
    try {
      const jobCreateRes = await axios.post(
        `${API_URL}/detect/yolo/mock`,
        {},
        {
          headers: { Authorization: `Bearer ${token}` },
          timeout: 10000
        }
      );
      jobId = jobCreateRes.data.jobId;
      console.log('✅ Job created:', jobId);
    } catch (err) {
      console.log('❌ Job creation failed:', err.message);
    }

    // Test 4: Check job status endpoint
    if (jobId) {
      console.log('\n4️⃣ Testing job status endpoint...');
      try {
        const jobRes = await axios.get(
          `${API_URL}/detect/yolo/job/${jobId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
            timeout: 10000
          }
        );
        console.log('✅ Job status endpoint working');
        console.log('   Status:', jobRes.data.status);
        console.log('   Progress:', jobRes.data.progress);
        console.log('   Message:', jobRes.data.message);
      } catch (err) {
        console.log('❌ Job status endpoint error:', err.message);
      }
    }

    // Test 5: Test CORS headers
    console.log('\n5️⃣ Testing CORS headers...');
    try {
      const corsRes = await axios.options(
        `${API_URL}/detect/yolo/mock`,
        {
          headers: {
            'Origin': 'http://localhost:3001',
            'Access-Control-Request-Method': 'POST'
          },
          timeout: 10000
        }
      );
      console.log('✅ CORS preflight successful');
      console.log('   CORS headers:', corsRes.headers);
    } catch (err) {
      console.log('⚠️ CORS preflight:', err.message);
    }

    // Test 6: Check available routes
    console.log('\n6️⃣ Checking available routes...');
    const routes = [
      'GET /api/detect/yolo/job/:jobId',
      'POST /api/detect/yolo/start-job',
      'POST /api/detect/yolo/mock',
      'GET /api/detect/yolo/job/:jobId/stream'
    ];
    console.log('Expected routes:');
    routes.forEach(r => console.log(`   ${r}`));

    console.log('\n✅ All tests completed!');

  } catch (err) {
    console.error('❌ Test error:', err.message);
  }
}

// Run tests
testJobQueue().then(() => {
  console.log('\n==========================================');
  console.log('✅ Test suite finished');
  console.log('==========================================');
  process.exit(0);
}).catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
