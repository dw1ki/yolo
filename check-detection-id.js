const mongoose = require('mongoose');
const Detection = require('./src/models/Detection');
require('dotenv').config();

mongoose.connect(process.env.MONGO_URI).then(() => {
  // Get the latest detection where id is 698e9aa2f70ae75d7f245b59
  Detection.findById('698e9aa2f70ae75d7f245b59').lean().then(detection => {
    if (!detection) {
      console.log('Detection not found');
    } else {
      console.log('\n=== DETECTION RESULTS ===');
      console.log('ID:', detection._id);
      console.log('Status:', detection.status);
      console.log('yoloResults.outputVideoUrl:', detection.yoloResults?.outputVideoUrl);
      console.log('videoUrl:', detection.videoUrl);
      console.log('yoloResults keys:', Object.keys(detection.yoloResults || {}));
      console.log('\nFull yoloResults:');
      console.log(JSON.stringify(detection.yoloResults, null, 2));
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
