FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies untuk OpenCV & FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install NumPy 1.x (YOLOv8 butuh ini)
# 2. Install Torch CPU (Hemat size)
RUN pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install requirements + multipart (WAJIB untuk upload file)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-multipart

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]