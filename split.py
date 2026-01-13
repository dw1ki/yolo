import os
import random
import shutil

BASE = "data"
IMG_TRAIN = f"{BASE}/images/train"
IMG_VAL = f"{BASE}/images/val"
LBL_TRAIN = f"{BASE}/labels/train"
LBL_VAL = f"{BASE}/labels/val"

os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

files = [f for f in os.listdir(IMG_TRAIN) if f.endswith(".jpg")]
random.shuffle(files)

val_ratio = 0.2
val_count = int(len(files) * val_ratio)

for f in files[:val_count]:
    shutil.move(f"{IMG_TRAIN}/{f}", f"{IMG_VAL}/{f}")
    txt = f.replace(".jpg", ".txt")
    if os.path.exists(f"{LBL_TRAIN}/{txt}"):
        shutil.move(f"{LBL_TRAIN}/{txt}", f"{LBL_VAL}/{txt}")

print("Split selesai")
