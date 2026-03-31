const mongoose = require('mongoose');
const Detection = require('./src/models/Detection');
require('dotenv').config();

mongoose.connect(process.env.MONGO_URI).then(() => {
  Detection.findOne().sort({createdAt: -1}).lean().then(detection => {
    if (!detection) {
      console.log('No detections found');
    } else {
      console.log('\n=== LATEST DETECTION ===');
      console.log('ID:', detection._id);
      console.log('Status:', detection.status);
      console.log('outputVideoUrl:', detection.yoloResults?.outputVideoUrl);
      console.log('videoUrl:', detection.videoUrl);
      console.log('cloudinaryUrl:', detection.yoloResults?.cloudinaryUrl);
      console.log('totalVehicles:', detection.yoloResults?.totalVehicles);
    }
    process.exit(0);
  }).catch(err => {
    console.error('Query error:', err.message);
    process.exit(1);
  });
}).catch(err => {
  console.error('Connection error:', err.message);
  process.exit(1);
});
