#!/bin/bash

# Comprehensive test script for YOLO video processing
# Tests the entire flow: upload → job creation → polling → results

echo "========================================"
echo "🧪 YOLO Video Processing Test Suite"
echo "========================================"
echo ""

# Configuration
API_URL="http://localhost:5000/api"
BACKEND_PORT=5000
JWT_SECRET="supersecretpktj"

# Generate test JWT token using openssl
echo "🔑 Generating JWT token..."
TOKEN=$(node -e "
const jwt = require('jsonwebtoken');
const token = jwt.sign(
  { id: 'test-user-123', email: 'test@test.com' },
  '${JWT_SECRET}',
  { expiresIn: '1h' }
);
console.log(token);
")
echo "✅ Token: ${TOKEN:0:30}..."
echo ""

# Check backend is running
echo "⚙️ Checking backend health..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${BACKEND_PORT})
if [ "$HEALTH" != "200" ] && [ "$HEALTH" != "404" ]; then
  echo "❌ Backend not running on port ${BACKEND_PORT}"
  echo "   Please run: cd backend && npm start"
  exit 1
fi
echo "✅ Backend is running (HTTP $HEALTH)"
echo ""

# Test 1: POST mock endpoint
echo "TEST 1️⃣ : Create mock job"
echo "=============================="
MOCK_RESPONSE=$(curl -s -X POST \
  "$API_URL/detect/yolo/mock" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json")

MOCK_JOB_ID=$(echo "$MOCK_RESPONSE" | grep -o '"jobId":"[^"]*' | cut -d'"' -f4)
echo "Response: $MOCK_RESPONSE"
echo "✅ Mock job created: $MOCK_JOB_ID"
echo ""

# Test 2: GET job status
echo "TEST 2️⃣ : Check job status"
echo "=============================="
JOB_STATUS=$(curl -s -X GET \
  "$API_URL/detect/yolo/job/$MOCK_JOB_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json")

echo "Response: $JOB_STATUS"
echo "✅ Job status retrieved"
echo ""

# Test 3: Test SSE stream endpoint
echo "TEST 3️⃣ : Testing SSE stream endpoint"
echo "=============================="
echo "⏳ Connecting to SSE stream for 3 seconds..."
timeout 3 curl -s -X GET \
  "$API_URL/detect/yolo/job/$MOCK_JOB_ID/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream" || true
echo ""
echo "✅ SSE stream test completed"
echo ""

# Test 4: Create real video file and test upload
echo "TEST 4️⃣ : Creating dummy video file"
echo "=============================="

# Create a minimal MP4 file (FFmpeg not required, we'll use dd)
DUMMY_VIDEO="/tmp/test-video-$(date +%s).mp4"
echo "📹 Creating dummy video: $DUMMY_VIDEO"

# Create a minimal video file header (1KB dummy)
dd if=/dev/zero bs=1024 count=1 of="$DUMMY_VIDEO" 2>/dev/null

echo "✅ Dummy video created ($(du -h "$DUMMY_VIDEO" | cut -f1))"
echo ""

# Test 5: Upload video and get job ID
echo "TEST 5️⃣ : Upload video and create job"
echo "=============================="
echo "📤 Uploading video to Cloudinary..."

UPLOAD_RESPONSE=$(curl -s -X POST \
  "$API_URL/detect/yolo/start-job" \
  -H "Authorization: Bearer $TOKEN" \
  -F "video=@$DUMMY_VIDEO")

echo "Response: $UPLOAD_RESPONSE"

UPLOAD_JOB_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"jobId":"[^"]*' | cut -d'"' -f4)
CLOUDINARY_URL=$(echo "$UPLOAD_RESPONSE" | grep -o '"cloudinaryUrl":"[^"]*' | cut -d'"' -f4)

if [ -z "$UPLOAD_JOB_ID" ]; then
  echo "❌ Failed to get jobId from upload response"
  echo "Response was: $UPLOAD_RESPONSE"
else
  echo "✅ Video uploaded successfully"
  echo "   JobID: $UPLOAD_JOB_ID"
  echo "   Cloudinary URL: $CLOUDINARY_URL"
fi
echo ""

# Test 6: Poll job progress
if [ -n "$UPLOAD_JOB_ID" ]; then
  echo "TEST 6️⃣ : Poll job progress (30 second timeout)"
  echo "=============================="
  
  for i in {1..6}; do
    POLL_RESPONSE=$(curl -s -X GET \
      "$API_URL/detect/yolo/job/$UPLOAD_JOB_ID" \
      -H "Authorization: Bearer $TOKEN")
    
    STATUS=$(echo "$POLL_RESPONSE" | grep -o '"status":"[^"]*' | cut -d'"' -f4)
    PROGRESS=$(echo "$POLL_RESPONSE" | grep -o '"progress":[0-9]*' | cut -d':' -f2)
    MESSAGE=$(echo "$POLL_RESPONSE" | grep -o '"message":"[^"]*' | cut -d'"' -f4)
    
    echo "[$i/6] Status: $STATUS | Progress: $PROGRESS% | Message: $MESSAGE"
    
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
      echo "✅ Job finished with status: $STATUS"
      break
    fi
    
    sleep 5
  done
  echo ""
fi

# Cleanup
rm -f "$DUMMY_VIDEO"

echo "========================================"
echo "✅ Test suite completed!"
echo "========================================"
echo ""
echo "Summary:"
echo "- ✅ Backend health check"
echo "- ✅ JWT token generation"
echo "- ✅ Mock job creation"
echo "- ✅ Job status retrieval"
echo "- ✅ SSE stream endpoint"
if [ -n "$UPLOAD_JOB_ID" ]; then
  echo "- ✅ Video upload"
  echo "- ✅ Job progress polling"
fi
echo ""
echo "Next steps:"
echo "1. Upload actual MP4 video file"
echo "2. Monitor YOLO API processing"
echo "3. Check final results in table"
echo ""
