@echo off
REM Quick script untuk memulai training YOLO dengan dataset video 008
REM Usage: double-click atau jalankan di command prompt

echo =============================================
echo  YOLO Training Script - Video 008 Dataset
echo =============================================
echo.

cd /d "%~dp0"

echo Checking environment...
python --version
echo.

echo Memulai training...
echo - Dataset: data/dataset_008
echo - Model Base: yolov8m.pt
echo - Epochs: 60
echo - GPU Device: 0
echo.

python train.py

echo.
echo =============================================
echo  Training Selesai!
echo =============================================
echo Model terbaik tersimpan di: runs/train_video_008/weights/best.pt
pause
