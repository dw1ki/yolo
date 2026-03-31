import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Detection from './src/models/Detection.js';

dotenv.config();

async function normalizeVehicleTypes() {
  try {
    console.log('🚀 Normalizing vehicle types...');
    
    await mongoose.connect(process.env.MONGO_URI);
    console.log('✅ Connected to MongoDB');
    
    const detections = await Detection.find({});
    console.log(`📊 Found ${detections.length} detection records`);
    
    let updatedCount = 0;
    
    for (const detection of detections) {
      const vt = detection.yoloResults?.vehicleTypes || {};
      
      // Check if already normalized (has mobil, bus, truk fields)
      if (vt.mobil !== undefined && vt.bus !== undefined && vt.truk !== undefined) {
        console.log(`⏭️  Skipping ${detection.fileName} - already normalized`);
        continue;
      }
      
      console.log(`\n📝 Processing: ${detection.fileName}`);
      console.log(`   Old format:`, vt);
      
      // Combine old format into new format
      const mobil = vt.mobilPenumpang || 0;
      const bus = vt.bus || 0;
      const truk = (vt.truckRingan || 0) + (vt.truckBerat || 0);
      
      detection.yoloResults.vehicleTypes = {
        mobil,
        bus,
        truk
      };
      
      await detection.save();
      updatedCount++;
      
      console.log(`   ✅ Normalized to:`, { mobil, bus, truk });
    }
    
    console.log(`\n✅ Done! Normalized ${updatedCount} records`);
    process.exit(0);
  } catch (err) {
    console.error('❌ Error:', err.message);
    process.exit(1);
  }
}

normalizeVehicleTypes();
