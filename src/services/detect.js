import axios from "axios";

export const detectVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append("video", videoFile);

  const res = await axios.post(
    "http://localhost:5000/api/detect",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 0, // YOLO lama
    }
  );

  return res.data;
};
