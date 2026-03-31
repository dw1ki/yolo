import dotenv from "dotenv";
import mongoose from "mongoose";
import app from "./app.js";

dotenv.config();

const PORT = process.env.PORT || 5000;

console.log("\n" + "=".repeat(70));
console.log("🚀 [SERVER] Starting PKTJ Backend Server");
console.log("=".repeat(70));
console.log(`📅 Deployment Timestamp: ${new Date().toISOString()}`);
console.log(`🌍 Node Environment: ${process.env.NODE_ENV || 'development'}`);
console.log(`🔧 Vercel Runtime: ${process.env.VERCEL_RUNTIME || 'N/A'}`);
console.log(`📍 Current Working Directory: ${process.cwd()}`);
console.log("=".repeat(70) + "\n");

// Start server immediately
const server = app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});

// ==================== TIMEOUT CONFIGURATION FOR LONG-RUNNING JOBS ====================
// Set higher timeout untuk video processing yang panjang (1+ jam)
server.timeout = 15 * 60 * 1000; // 15 menit total timeout
server.keepAliveTimeout = 65 * 1000; // 65 detik keep-alive (default 5 menit untuk Production)
server.requestTimeout = 12 * 60 * 1000; // 12 menit untuk individual request

// Adjust socket timeout
server.on('connection', (socket) => {
  socket.setTimeout(15 * 60 * 1000); // 15 menit socket timeout
  socket.setKeepAlive(true);
});

// Handle server errors
server.on('error', (err) => {
  console.error("❌ Server error:", err);
  process.exit(1);
});

// ==================== Global Error Handlers ====================
// Prevent unhandled promise rejections from crashing the server
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ [UNHANDLED REJECTION]', reason);
  console.error('   Promise:', promise);
  // Don't crash - just log it
});

// Prevent uncaught exceptions from crashing the server
process.on('uncaughtException', (err) => {
  console.error('❌ [UNCAUGHT EXCEPTION]', err);
  // Try to send one final response if possible, but don't crash
});

// Connect MongoDB in background (don't block server startup)
mongoose
  .connect(process.env.MONGO_URI, {
    connectTimeoutMS: 10000,
    socketTimeoutMS: 45000,
  })
  .then(() => {
    console.log("✅ MongoDB connected");
  })
  .catch((err) => {
    console.error("⚠️ MongoDB connection error (server still running):", err.message);
  });
