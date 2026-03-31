import mongoose from "mongoose";

const detectionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: [true, "User ID wajib diisi"],
    },

    // ==================== VIDEO INFORMATION ====================
    videoUrl: {
      type: String,
      required: [true, "Video URL wajib diisi"],
    },
    cloudinaryPublicId: {
      type: String,
      required: false,
    },
    cloudinarySecureUrl: {
      type: String,
      required: false,
    },
    storageType: {
      type: String,
      enum: ["cloudinary", "local"],
      default: "local",
    },
    fileName: {
      type: String,
    },
    fileSize: {
      type: Number,
    },
    videoDuration: {
      type: Number, // in seconds
      default: 0,
    },
    recordingInterval: {
      type: String, // e.g., "08:00 - 09:00"
      default: "",
    },

    // ==================== YOLO DETECTION RESULTS ====================
    yoloResults: {
      totalVehicles: {
        type: Number,
        default: 0,
      },
      vehicleTypes: {
        mobil: { type: Number, default: 0 }, // Mobil penumpang (× 1.0 SMP)
        bus: { type: Number, default: 0 }, // Bus (× 1.3 SMP)
        truk: { type: Number, default: 0 }, // Truk (ringan + berat, × 1.2-2.0 SMP)
      },
      volumeSMP: {
        type: Number, // Total SMP from vehicle counting
        default: 0,
      },
      avgConfidence: {
        type: Number,
        default: 0,
      },
      totalFrames: {
        type: Number,
        default: 0,
      },
      rawData: mongoose.Schema.Types.Mixed,
      // NEW: Per-lane vehicle breakdown
      leftLaneCount: {
        type: Number,
        default: 0,
      },
      rightLaneCount: {
        type: Number,
        default: 0,
      },
      leftLane: {
        mobil: { type: Number, default: 0 },
        bus: { type: Number, default: 0 },
        truk: { type: Number, default: 0 },
      },
      rightLane: {
        mobil: { type: Number, default: 0 },
        bus: { type: Number, default: 0 },
        truk: { type: Number, default: 0 },
      },
    },

    // ==================== ROAD PARAMETERS ====================
    roadParameters: {
      roadName: {
        type: String,
        default: "MBZ",
      },
      roadType: {
        type: String,
        default: "4/2 D", // 4 lajur, 2 arah
      },
      numLanes: {
        type: Number,
        default: 4, // n = 4
      },
      baseCapacity: {
        type: Number,
        default: 5000, // C0 = 5000 smp/jam per lajur
      },
      laneWidth: {
        type: Number,
        default: 3.5, // Lebar lajur dalam meter
      },
      baseSpeed: {
        type: Number,
        default: 88, // MP (kecepatan dasar) km/jam
      },
      effectiveWidthFactor: {
        type: Number,
        default: 1.0, // FCLE (Faktor Lebar Efektif)
      },
    },

    // ==================== CALCULATION RESULTS ====================
    // Formula: C = n × C0 × FCLE
    // DJ = Q / C
    // LOS = Determine based on DJ
    calculations: {
      totalSMP: {
        type: Number,
        default: 0,
      },
      capacity: {
        type: Number, // C = n × C0 × FCLE (smp/jam)
        default: 0,
      },
      formula: {
        n: Number,
        C0: Number,
        FCLE: Number,
        equation: String, // e.g., "4 × 5000 × 1.0 = 20000"
      },
      volume: {
        type: Number, // Q = Volume dari YOLO (smp/jam)
        default: 0,
      },
      degree: {
        type: Number, // DJ = Q / C (0-1+)
        default: 0,
      },
      degreeFormula: String, // e.g., "1500 / 20000 = 0.075"
      los: {
        type: String, // A, B, C, D, E, F
        enum: ["A", "B", "C", "D", "E", "F", ""],
        default: "",
      },
      losCategory: {
        type: String, // Lancar, Stabil, Macet, dll
        default: "",
      },
      losDescription: String,
    },

    // ==================== AUTO-GENERATED CONCLUSION ====================
    conclusion: {
      type: String,
      default: "",
    },

    // ==================== STATUS & VERIFICATION ====================
    status: {
      type: String,
      enum: ["draft", "processing", "completed", "verified", "failed"],
      default: "draft",
    },
    error: {
      type: String,
    },

    // ==================== METADATA ====================
    createdBy: String, // User name
    verifiedAt: Date,
    verifiedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
    },
    notes: String,
  },
  {
    timestamps: true,
  }
);

// Indexes untuk query yang sering
detectionSchema.index({ userId: 1, createdAt: -1 });
detectionSchema.index({ status: 1 });
detectionSchema.index({ "calculations.los": 1 });

export default mongoose.model("Detection", detectionSchema);

