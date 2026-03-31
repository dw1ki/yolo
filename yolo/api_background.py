import cloudinary
import cloudinary.uploader
import os

# Ambil dari environment variable atau isi langsung untuk testing
cloudinary.config( 
  cloud_name = "dgiv24lgt", 
  api_key = "789599361888426", 
  api_secret = "Bu-A3N8Vbkep05qe40f-NifJxmw",
  secure = True
)

def upload_result(file_path):
    response = cloudinary.uploader.upload(file_path, resource_type="video")
    return response['secure_url']