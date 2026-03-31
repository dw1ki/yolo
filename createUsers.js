import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import dotenv from "dotenv";
import User from "./src/models/User.js"; // pastikan path sesuai

dotenv.config();

// Connect MongoDB
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.log(err));

async function createUsers() {
  try {
    // Hash password
    const adminPassword = await bcrypt.hash("admin123", 10);
    const surveyorPassword = await bcrypt.hash("surveyor123", 10);
    const userPassword = await bcrypt.hash("user123", 10);

    // Users
    const users = [
      { name: "Admin PKTJ", email: "admin@pktj.com", role: "admin", password: adminPassword },
      { name: "Surveyor PKTJ", email: "surveyor@pktj.com", role: "surveyor", password: surveyorPassword },
      { name: "User PKTJ", email: "user@pktj.com", role: "user", password: userPassword },
    ];

    for (const u of users) {
      const exists = await User.findOne({ email: u.email });
      if (!exists) {
        await User.create(u);
        console.log(`User created: ${u.email} (${u.role})`);
      } else {
        console.log(`User already exists: ${u.email}`);
      }
    }

    console.log("Done!");
    process.exit(0);
  } catch (err) {
    console.log(err);
    process.exit(1);
  }
}

// Jalankan fungsi
createUsers();
