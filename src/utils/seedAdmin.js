import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import dotenv from "dotenv";
import User from "../models/User.js";

dotenv.config();

const seedAdmin = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB");

    // Check if admin already exists
    const existingAdmin = await User.findOne({ email: "admin@example.com" });
    if (existingAdmin) {
      console.log("Admin user sudah ada!");
      await mongoose.disconnect();
      process.exit(0);
    }

    // DO NOT hash password here - the pre-save hook will do it
    const admin = new User({
      name: "Yunindra Eka Ariffansyah",
      email: "admin@example.com",
      password: "admin123", // Plain password - will be hashed in pre-save hook
      role: "admin"
    });

    const savedAdmin = await admin.save();
    console.log("Admin user created successfully!", savedAdmin._id);
    
    await mongoose.disconnect();
    process.exit(0);
  } catch (err) {
    console.error("Error creating admin:", err.message);
    await mongoose.disconnect();
    process.exit(1);
  }
};

seedAdmin();
