#!/usr/bin/env node

/**
 * End-to-end test untuk verifikasi video detection flow
 * - Backend: Railway
 * - Frontend: Vercel
 * - YOLO API: localhost:8000 (via ngrok)
 */

const http = require('http');
const https = require('https');
const fs = require('fs');

const BACKEND_URL = process.env.BACKEND_URL || 'https://backend-production-5fbe.up.railway.app';
const PYTHON_API = process.env.PYTHON_API || 'https://hurtling-unforecasted-horace.ngrok-free.dev';

console.log('\n🧪 E2E Test: Video Detection Flow\n');
console.log(`Backend:    ${BACKEND_URL}`);
console.log(`YOLO API:   ${PYTHON_API}`);
console.log(`\n`);

// Test 1: Verify Python API is accessible
console.log('📝 Test 1: Check Python API health...');
const testUrl = `${PYTHON_API}/result/test_job`;
const protocol = testUrl.startsWith('https') ? https : http;

protocol.get(testUrl, {
  rejectUnauthorized: false,
  timeout: 10000
}, (res) => {
  console.log(`✅ Python API responded with status: ${res.statusCode}`);
  
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    try {
      const json = JSON.parse(data);
      console.log(`✅ Response has outputVideoUrl: ${json.outputVideoUrl ? '✓' : '✗'}`);
      console.log(`✅ Response has backendUrl: ${json.backendUrl ? '✓' : '✗'}`);
      console.log(`\n📊 Sample response:`);
      console.log(JSON.stringify({
        status: json.status,
        outputVideoUrl: json.outputVideoUrl,
        backendUrl: json.backendUrl,
        progress: json.progress
      }, null, 2));
    } catch (e) {
      console.log(`Response: ${data.substring(0, 200)}`);
    }
  });
}).on('error', (err) => {
  console.error(`❌ Error: ${err.message}`);
});
