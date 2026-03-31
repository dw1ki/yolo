import cloudinary.uploader

def upload_video_to_cloudinary(video_path: str):
    result = cloudinary.uploader.upload(
        video_path,
        resource_type="video",
        folder="yolo_results"
    )
    return result["secure_url"]
