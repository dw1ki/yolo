/**
 * ========================================
 * RUMUS PERHITUNGAN KAPASITAS JALAN TOL
 * PKJI 2023 (Panduan Kapasitas Jalan Indonesia)
 * ========================================
 * 
 * FORMULA KAPASITAS:
 * C = n × C0 × FCLE
 * 
 * Keterangan:
 * - C    : Kapasitas jalur lalu lintas (SMP/jam)
 * - n    : Jumlah lajur
 * - C0   : Kapasitas dasar per lajur (5000 SMP/jam untuk jalan tol)
 * - FCLE : Faktor Lebar Efektif Jalur
 * 
 * DERAJAT KEJENUHAN:
 * DJ = Q / C
 * - DJ : Derajat Kejenuhan (0-1+)
 * - Q  : Volume lalu lintas (SMP/jam)
 * - C  : Kapasitas (SMP/jam)
 * 
 * LEVEL OF SERVICE (LOS):
 * A (DJ < 0.60) : Lancar, kecepatan tinggi
 * B (0.60-0.70) : Lancar, kecepatan mulai dibatasi
 * C (0.70-0.80) : Stabil, kecepatan dibatasi
 * D (0.80-0.90) : Mendekati tidak stabil
 * E (0.90-1.00) : Tidak stabil
 * F (DJ ≥ 1.00) : Terhambat (macet)
 */

// ==========================================
// 1. SMP (SATUAN MOBIL PENUMPANG) CONVERSION
// ==========================================
/**
 * Konversi dari tipe kendaraan ke SMP
 * - Mobil penumpang: 1.0 SMP
 * - Truck ringan (< 5 ton): 1.2 SMP
 * - Truck berat (>= 5 ton): 2.0 SMP
 * - Bus: 1.3 SMP
 */
const SMP_FACTORS = {
  mobilPenumpang: 1.0,
  truckRingan: 1.2,
  truckBerat: 2.0,
  bus: 1.3
};

/**
 * Convert vehicle counts to SMP (Satuan Mobil Penumpang)
 * @param {Object} vehicleTypes - { mobilPenumpang, truckRingan, truckBerat, bus }
 * @returns {number} Total SMP
 */
const convertToSMP = (vehicleTypes) => {
  let totalSMP = 0;
  
  for (const [type, count] of Object.entries(vehicleTypes)) {
    if (SMP_FACTORS[type]) {
      totalSMP += count * SMP_FACTORS[type];
    }
  }
  
  return Math.round(totalSMP * 100) / 100;
};

// ==========================================
// 2. CAPACITY CALCULATION
// ==========================================
/**
 * Calculate capacity using formula: C = n × C0 × FCLE
 * @param {number} n - Number of lanes
 * @param {number} C0 - Base capacity (5000 SMP/jam for toll roads)
 * @param {number} FCLE - Effective width factor (default 1.0)
 * @returns {number} Total capacity in SMP/jam
 */
const calculateCapacity = (n, C0 = 5000, FCLE = 1.0) => {
  // If FCLE is 0 or not provided, use default value of 1.0
  const fcle = FCLE === 0 || FCLE === null ? 1.0 : FCLE;
  const capacity = n * C0 * fcle;
  
  return Math.round(capacity);
};

// ==========================================
// 3. VOLUME CALCULATION
// ==========================================
/**
 * Calculate volume in SMP/jam from YOLO results
 * Q = (Total SMP / Video Duration in hours)
 * @param {number} totalSMP - Total SMP from vehicle counting
 * @param {number} durationSeconds - Video duration in seconds
 * @returns {number} Volume in SMP/jam
 */
const calculateVolume = (totalSMP, durationSeconds) => {
  if (!durationSeconds || durationSeconds <= 0) {
    return 0;
  }
  
  const durationHours = durationSeconds / 3600;
  const volume = totalSMP / durationHours;
  
  return Math.round(volume);
};

// ==========================================
// 4. DEGREE OF SATURATION (DJ)
// ==========================================
/**
 * Calculate Degree of Saturation: DJ = Q / C
 * @param {number} volume - Traffic volume in SMP/jam
 * @param {number} capacity - Road capacity in SMP/jam
 * @returns {number} DJ value (0-1+)
 */
const calculateDegreeOfSaturation = (volume, capacity) => {
  if (!capacity || capacity <= 0) {
    return volume > 0 ? 999 : 0; // Return very high value if capacity is 0
  }
  
  const dj = volume / capacity;
  return Math.round(dj * 1000) / 1000; // Round to 3 decimals
};

// ==========================================
// 5. LEVEL OF SERVICE (LOS) DETERMINATION
// ==========================================
/**
 * Determine Level of Service based on DJ
 * @param {number} dj - Degree of Saturation value
 * @returns {Object} { los, category, description }
 */
const determineLOS = (dj) => {
  let los, category, description;
  
  if (dj < 0.60) {
    los = 'A';
    category = 'Lancar';
    description = 'Kondisi lalu lintas sangat lancar, kecepatan tinggi, pengemudi bebas memilih kecepatan';
  } else if (dj < 0.70) {
    los = 'B';
    category = 'Lancar';
    description = 'Kondisi lalu lintas lancar, kecepatan mulai dibatasi oleh kepadatan lalu lintas';
  } else if (dj < 0.80) {
    los = 'C';
    category = 'Stabil';
    description = 'Kondisi lalu lintas stabil, kecepatan sudah dibatasi, pengemudi memiliki keterbatasan dalam bergerak';
  } else if (dj < 0.90) {
    los = 'D';
    category = 'Mendekati Tidak Stabil';
    description = 'Kondisi lalu lintas mendekati tidak stabil, kecepatan tergantung pada kepadatan lalu lintas';
  } else if (dj < 1.00) {
    los = 'E';
    category = 'Tidak Stabil';
    description = 'Kondisi lalu lintas tidak stabil, volume sama dengan kapasitas, arus lalu lintas terganggu';
  } else {
    los = 'F';
    category = 'Terhambat (Macet)';
    description = 'Kondisi lalu lintas terhambat, volume melebihi kapasitas, terjadi kemacetan';
  }
  
  return { los, category, description };
};

// ==========================================
// 6. AUTO-GENERATE CONCLUSION
// ==========================================
/**
 * Generate conclusion text based on calculation results
 * @param {Object} params - Calculation parameters
 * @returns {string} Conclusion text in Indonesian
 */
const generateConclusion = ({
  roadName = 'Ruas Jalan',
  interval = '08:00 - 09:00',
  totalVehicles = 0,
  dj = 0,
  los = 'F',
  volume = 0,
  capacity = 0,
  losCategory = 'Terhambat'
}) => {
  const losDescription = {
    'A': 'lancar dengan kecepatan tinggi',
    'B': 'lancar dengan kecepatan terbatas',
    'C': 'stabil dengan kecepatan terbatas',
    'D': 'mendekati tidak stabil',
    'E': 'tidak stabil',
    'F': 'terhambat dengan kemacetan'
  };
  
  const description = losDescription[los] || losCategory.toLowerCase();
  
  return `Berdasarkan hasil analisis kinerja lalu lintas pada Ruas ${roadName} yang dilakukan pada pukul ${interval}, diperoleh data sebagai berikut: total kendaraan terdeteksi sebanyak ${totalVehicles} unit dengan volume lalu lintas sebesar ${volume} SMP/jam. Dengan kapasitas jalan ${capacity} SMP/jam, ruas jalan ini memiliki Derajat Kejenuhan (DJ) sebesar ${dj}. Kondisi lalu lintas ruas ini menunjukkan Level of Service (LOS) ${los} yang artinya ${description}. Rekomendasi: ${los === 'F' ? 'Diperlukan tindakan perbaikan segera untuk mengurangi kemacetan' : los === 'E' ? 'Perlu monitoring ketat dan persiapan manajemen lalu lintas' : 'Kondisi lalu lintas dapat ditoleransi'}.`;
};

// ==========================================
// 7. MAIN CALCULATION FUNCTION
// ==========================================
/**
 * Perform complete calculation
 * @param {Object} detectionData - Complete detection data with YOLO results & road parameters
 * @returns {Object} Complete calculation results
 */
const performCalculation = (detectionData) => {
  const {
    yoloResults = {},
    roadParameters = {},
    videoDuration = 0,
    recordingInterval = '',
    roadName = 'Ruas Jalan'
  } = detectionData;

  // Step 1: Convert vehicles to SMP
  const vehicleTypes = yoloResults.vehicleTypes || {
    mobilPenumpang: 0,
    truckRingan: 0,
    truckBerat: 0,
    bus: 0
  };
  
  const totalSMP = convertToSMP(vehicleTypes);

  // Step 2: Calculate Volume (Q) in SMP/jam
  const volume = calculateVolume(totalSMP, videoDuration);

  // Step 3: Calculate Capacity (C) = n × C0 × FCLE
  const numLanes = roadParameters.numLanes || 4;
  const baseCapacity = roadParameters.baseCapacity || 5000;
  const effectiveWidthFactor = roadParameters.effectiveWidthFactor || 1.0;
  
  const capacity = calculateCapacity(numLanes, baseCapacity, effectiveWidthFactor);

  // Step 4: Calculate DJ = Q / C
  const dj = calculateDegreeOfSaturation(volume, capacity);

  // Step 5: Determine LOS
  const { los, category, description } = determineLOS(dj);

  // Step 6: Generate conclusion
  const conclusion = generateConclusion({
    roadName: roadParameters.roadName || roadName,
    interval: recordingInterval,
    totalVehicles: yoloResults.totalVehicles || 0,
    dj,
    los,
    volume,
    capacity,
    losCategory: category
  });

  return {
    // Input data summary
    totalVehicles: yoloResults.totalVehicles || 0,
    totalSMP,
    
    // Capacity calculation
    capacity,
    formula: {
      n: numLanes,
      C0: baseCapacity,
      FCLE: effectiveWidthFactor,
      equation: `${numLanes} × ${baseCapacity} × ${effectiveWidthFactor} = ${capacity}`
    },
    
    // Volume calculation
    volume,
    
    // Degree of saturation
    degree: dj,
    degreeFormula: `${volume} / ${capacity} = ${dj}`,
    
    // Level of service
    los,
    losCategory: category,
    losDescription: description,
    
    // Conclusion
    conclusion,
    
    // Metadata
    calculatedAt: new Date(),
    status: 'completed'
  };
};

// ==========================================
// 8. VALIDATION FUNCTIONS
// ==========================================

/**
 * Validate road parameters before calculation
 */
const validateRoadParameters = (roadParams) => {
  const errors = [];
  
  if (!roadParams.numLanes || roadParams.numLanes < 1) {
    errors.push('Jumlah lajur harus minimal 1');
  }
  
  if (!roadParams.baseCapacity || roadParams.baseCapacity < 1000) {
    errors.push('Kapasitas dasar harus minimal 1000');
  }
  
  if (roadParams.effectiveWidthFactor !== 0 && (!roadParams.effectiveWidthFactor || roadParams.effectiveWidthFactor < 0)) {
    errors.push('Faktor lebar efektif tidak valid');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
};

/**
 * Validate YOLO results
 */
const validateYOLOResults = (yoloResults) => {
  const errors = [];
  
  if (!yoloResults || typeof yoloResults !== 'object') {
    errors.push('Data YOLO tidak valid');
  }
  
  if (!yoloResults.vehicleTypes) {
    errors.push('Data tipe kendaraan tidak ditemukan');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
};

// ==========================================
// EXPORTS
// ==========================================
export {
  // Constants
  SMP_FACTORS,
  
  // Conversion
  convertToSMP,
  
  // Calculations
  calculateCapacity,
  calculateVolume,
  calculateDegreeOfSaturation,
  
  // LOS & Conclusion
  determineLOS,
  generateConclusion,
  
  // Main function
  performCalculation,
  
  // Validation
  validateRoadParameters,
  validateYOLOResults
};
