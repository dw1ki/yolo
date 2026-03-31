import mongoose from 'mongoose';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Import Detection model
import Detection from './src/models/Detection.js';

async function migratePerLaneData() {
  try {
    console.log('🚀 Starting per-lane data migration...');
    
    // Connect to MongoDB
    const mongoUri = process.env.MONGO_URI || process.env.MONGODB_URI;
    console.log('📡 Connecting to MongoDB...');
    console.log('   URI:', mongoUri ? mongoUri.substring(0, 50) + '...' : 'NOT SET');
    
    await mongoose.connect(mongoUri);
    console.log('✅ Connected to MongoDB');
    
    // Find all detections
    const detections = await Detection.find({});
    console.log(`📊 Found ${detections.length} detection records`);
    
    let updatedCount = 0;
    
    for (const detection of detections) {
      // Check if already has PER-LANE DATA WITH ACTUAL VALUES (not just empty)
      const hasValidPerLane = 
        detection.yoloResults?.leftLane && 
        (detection.yoloResults.leftLane.mobil > 0 || 
         detection.yoloResults.leftLane.bus > 0 || 
         detection.yoloResults.leftLane.truk > 0);
      
      if (hasValidPerLane) {
        console.log(`⏭️  Skipping ${detection.fileName} - already has valid per-lane data`);
        continue;
      }
      
      console.log(`\n📝 Processing: ${detection.fileName} (ID: ${detection._id})`);
      
      // Extract vehicle counts from detections array or vehicleTypes
      let mobil = 0, bus = 0, truk = 0;
      
      if (detection.yoloResults?.rawData && Array.isArray(detection.yoloResults.rawData)) {
        detection.yoloResults.rawData.forEach(item => {
          if (item.type === 'Car' || item.class === 'mobil') mobil += item.count || 0;
          else if (item.type === 'Bus' || item.class === 'bus') bus += item.count || 0;
          else if (item.type === 'Truck' || item.class === 'truk') truk += item.count || 0;
        });
      }
      
      // Fallback to vehicleTypes - combine truckRingan + truckBerat into truk
      if (mobil === 0 && bus === 0 && truk === 0 && detection.yoloResults?.vehicleTypes) {
        mobil = detection.yoloResults.vehicleTypes.mobilPenumpang || 
                detection.yoloResults.vehicleTypes.mobil || 0;
        bus = detection.yoloResults.vehicleTypes.bus || 0;
        truk = (detection.yoloResults.vehicleTypes.truckRingan || 0) + 
               (detection.yoloResults.vehicleTypes.truckBerat || 0) ||
               detection.yoloResults.vehicleTypes.truk || 0;
      }
      
      const totalVehicles = detection.yoloResults?.totalVehicles || 0;
      
      console.log(`   Current breakdown: Mobil=${mobil}, Bus=${bus}, Truk=${truk}, Total=${totalVehicles}`);
      
      // Calculate per-lane split (55% left, 45% right)
      const leftRatio = 0.55;
      const rightRatio = 0.45;
      
      const leftLaneCount = Math.round(totalVehicles * leftRatio);
      const rightLaneCount = totalVehicles - leftLaneCount;
      
      const leftLaneMobil = Math.round(mobil * leftRatio);
      const leftLaneBus = Math.round(bus * leftRatio);
      const leftLaneTruk = Math.round(truk * leftRatio);
      
      const rightLaneMobil = mobil - leftLaneMobil;
      const rightLaneBus = bus - leftLaneBus;
      const rightLaneTruk = truk - leftLaneTruk;
      
      // Update yoloResults with per-lane data
      detection.yoloResults.leftLaneCount = leftLaneCount;
      detection.yoloResults.rightLaneCount = rightLaneCount;
      detection.yoloResults.leftLane = {
        mobil: leftLaneMobil,
        bus: leftLaneBus,
        truk: leftLaneTruk
      };
      detection.yoloResults.rightLane = {
        mobil: rightLaneMobil,
        bus: rightLaneBus,
        truk: rightLaneTruk
      };
      
      // ALSO: Normalize vehicleTypes to new format (mobil, bus, truk)
      detection.yoloResults.vehicleTypes = {
        mobil: mobil,
        bus: bus,
        truk: truk
      };
      
      // Save to database
      await detection.save();
      updatedCount++;
      
      console.log(`   ✅ Updated with per-lane data:`);
      console.log(`      Lajur Kiri: ${leftLaneCount} (M:${leftLaneMobil}, B:${leftLaneBus}, T:${leftLaneTruk})`);
      console.log(`      Lajur Kanan: ${rightLaneCount} (M:${rightLaneMobil}, B:${rightLaneBus}, T:${rightLaneTruk})`);
    }
    
    console.log(`\n✅ Migration complete! Updated ${updatedCount} records`);
    
    // Verify: Get updated data
    console.log('\n📊 Verification:');
    const updatedDetections = await Detection.find({});
    updatedDetections.forEach(det => {
      if (det.yoloResults?.leftLane) {
        console.log(`   ${det.fileName}: Kiri=${det.yoloResults.leftLaneCount}, Kanan=${det.yoloResults.rightLaneCount}`);
      }
    });
    
    process.exit(0);
  } catch (err) {
    console.error('❌ Migration error:', err.message);
    console.error('Stack:', err.stack);
    process.exit(1);
  }
}

migratePerLaneData();
