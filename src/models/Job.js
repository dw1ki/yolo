import mongoose from "mongoose";

const jobSchema = new mongoose.Schema(
  {
    videoUrl: String,
    result: Object
  },
  { timestamps: true }
);

export default mongoose.model("Job", jobSchema);
