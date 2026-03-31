import cv2
import os

VIDEO = "video/malam.mp4"
OUT = "data/images/train"

os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
frame_id = 0
save_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ambil tiap 10 frame
    if frame_id % 10 == 0:
        cv2.imwrite(f"{OUT}/frame_{save_id}.jpg", frame)
        save_id += 1

    frame_id += 1

cap.release()
print(f"Saved {save_id} frames")
