import mongoose from "mongoose";
import dotenv from "dotenv";
import User from "./src/models/User.js";

dotenv.config();

await mongoose.connect(process.env.MONGO_URI);

try {
  // Try to find existing admin
  let admin = await User.findOne({ email: "admin@example.com" });
  
  if (!admin) {
    // Create new admin if doesn't exist
    admin = new User({
      name: "Admin User",
      email: "admin@example.com",
      password: "admin123",
      role: "admin",
      isActive: true
    });
    await admin.save();
    console.log("✅ Admin account created!");
  } else {
    // Update existing admin password
    admin.password = "admin123";
    await admin.save();
    console.log("✅ Admin password reset!");
  }
  
  console.log(`Admin email: ${admin.email}`);
  console.log(`Admin role: ${admin.role}`);
  
} catch (err) {
  console.error("❌ Error:", err.message);
}

process.exit();
