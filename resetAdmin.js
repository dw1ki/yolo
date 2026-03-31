import mongoose from "mongoose";
import dotenv from "dotenv";
import User from "./src/models/User.js";

dotenv.config();

await mongoose.connect(process.env.MONGO_URI);

const admin = await User.findOne({ email: "admin@pktj.com" });

admin.password = "admin123"; // plaintext → akan di-hash oleh pre-save
await admin.save();

console.log("Password admin di-reset");

process.exit();
