"""
Merge dataset_008 and WhatsApp dataset into one combined dataset
"""
import shutil
import os
from pathlib import Path

combined_path = Path("D:/backup/pktj/backend/yolo/data/dataset_combined")
dataset_008_path = Path("D:/backup/pktj/backend/yolo/data/dataset_008")
whatsapp_images_path = Path("D:/backup/pktj/backend/yolo/data/images")
whatsapp_labels_path = Path("D:/backup/pktj/backend/yolo/data/labels")

print("🔄 Merging datasets...")

# Create directories
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    (combined_path / folder).mkdir(parents=True, exist_ok=True)

# Copy dataset_008
print("📋 Copying Video 008...")
for img in (dataset_008_path / "images" / "train").glob("*"):
    shutil.copy2(img, combined_path / "images" / "train")
print(f"   ✅ Train images: {len(list((combined_path / 'images' / 'train').glob('*')))}")

for img in (dataset_008_path / "images" / "val").glob("*"):
    shutil.copy2(img, combined_path / "images" / "val")

for lbl in (dataset_008_path / "labels" / "train").glob("*"):
    shutil.copy2(lbl, combined_path / "labels" / "train")

for lbl in (dataset_008_path / "labels" / "val").glob("*"):
    shutil.copy2(lbl, combined_path / "labels" / "val")

# Copy WhatsApp (need to rename to avoid conflicts)
print("📋 Copying WhatsApp...")
existing_008_count = len(list((combined_path / "images" / "train").glob("*")))

for i, img in enumerate((whatsapp_images_path / "train").glob("*.jpg")):
    new_name = f"whatsapp_{i}.jpg"
    shutil.copy2(img, combined_path / "images" / "train" / new_name)

for i, img in enumerate((whatsapp_images_path / "val").glob("*.jpg")):
    new_name = f"whatsapp_val_{i}.jpg"
    shutil.copy2(img, combined_path / "images" / "val" / new_name)

# Rename labels to match image names
whatsapp_label_files = list((whatsapp_labels_path / "train").glob("*.txt"))
for i, lbl in enumerate(whatsapp_label_files):
    new_name = f"whatsapp_{i}.txt"
    shutil.copy2(lbl, combined_path / "labels" / "train" / new_name)

whatsapp_label_files_val = list((whatsapp_labels_path / "val").glob("*.txt"))
for i, lbl in enumerate(whatsapp_label_files_val):
    new_name = f"whatsapp_val_{i}.txt"
    shutil.copy2(lbl, combined_path / "labels" / "val" / new_name)

# Count files
train_count = len(list((combined_path / "images" / "train").glob("*")))
val_count = len(list((combined_path / "images" / "val").glob("*")))

print(f"\n✅ Merge complete!")
print(f"   Training images: {train_count}")
print(f"   Validation images: {val_count}")
print(f"   Total: {train_count + val_count} images")
