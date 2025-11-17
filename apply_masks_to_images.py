import os
import cv2
import numpy as np
import glob
import shutil
import csv
from tqdm import tqdm
import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================
ORIGINAL_DIR = "data/"
MASKS_DIR = "output-sam/"
OUTPUT_DIR = "output-removed/"

# CSV log file (timestamped)
timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, f"log_{timestamp}.csv")

# Accepted extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Pixel intensity threshold (for white detection)
WHITE_THRESHOLD = 200

# ==========================================================
# PREPARE OUTPUT FOLDER
# ==========================================================
if os.path.exists(OUTPUT_DIR):
    print(f"🧹 Removing old output folder: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# PREPARE CSV LOGGER
# ==========================================================
os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
with open(LOG_CSV_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Index",
        "Original Image Path",
        "Mask Path",
        "Output Path",
        "Original Image Width",
        "Original Image Height",
        "Mask Width",
        "Mask Height",
        "White Pixels Count",
        "Pixels Replaced",
        "White Pixel Percentage",
        "Action"
    ])

# ==========================================================
# HELPER FUNCTION
# ==========================================================
def list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

# ==========================================================
# LOAD FILES
# ==========================================================
original_files = list_images(ORIGINAL_DIR)
mask_files = list_images(MASKS_DIR)

if not original_files:
    raise FileNotFoundError(f"No images found in ORIGINAL_DIR: {ORIGINAL_DIR}")
if not mask_files:
    raise FileNotFoundError(f"No images found in MASKS_DIR: {MASKS_DIR}")

if len(original_files) != len(mask_files):
    print(f"⚠️ Warning: Number of originals ({len(original_files)}) != masks ({len(mask_files)})")

print(f"Found {len(original_files)} original(s) and {len(mask_files)} mask(s)\n")

# ==========================================================
# PROCESS EACH PAIR
# ==========================================================
for i, (orig_path, mask_path) in enumerate(
    tqdm(zip(original_files, mask_files), total=min(len(original_files), len(mask_files)), desc="Processing")
):
    orig = cv2.imread(orig_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or mask is None:
        print(f"⚠️ Skipping {os.path.basename(orig_path)} due to read error")
        continue

    # Resize mask if needed
    if mask.shape[:2] != orig.shape[:2]:
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    img_h, img_w = orig.shape[:2]
    mask_h, mask_w = mask.shape[:2]

    # Identify white pixels
    white_pixels = mask > WHITE_THRESHOLD
    white_pixel_count = int(np.sum(white_pixels))
    total_pixels = img_h * img_w
    white_percentage = (white_pixel_count / total_pixels) * 100

    action = "Copied"
    replaced_pixels = 0

    if white_pixel_count > 0:
        # Blacken those pixels in the original
        orig[white_pixels] = [0, 0, 0]
        replaced_pixels = white_pixel_count
        action = "Modified"

    # Save processed image
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(orig_path))
    cv2.imwrite(output_path, orig)

    # Log entry
    with open(LOG_CSV_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            i + 1,
            os.path.abspath(orig_path),
            os.path.abspath(mask_path),
            os.path.abspath(output_path),
            img_w, img_h,
            mask_w, mask_h,
            white_pixel_count,
            replaced_pixels,
            f"{white_percentage:.4f}",
            action
        ])

# ==========================================================
# DONE
# ==========================================================
print("\n✅ Done!")
print(f"Processed images saved to: {OUTPUT_DIR}")
print(f"Detailed CSV log saved at: {LOG_CSV_PATH}")
