import os
import shutil
import torch
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# ==========================================================
# CONFIGURATION
# ==========================================================
HOME = ""

DATA_DIR = os.path.join(HOME, "data")
VOC_DIR = os.path.join(HOME, "output-filter/filtered_voc")
OUTPUT_SAM_DIR = os.path.join(HOME, "output-sam")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = os.path.join(HOME, "sam_vit_h_4b8939.pth")

# ==========================================================
# SETUP OUTPUT FOLDER
# ==========================================================
if os.path.exists(OUTPUT_SAM_DIR):
    print(f"🧹 Removing old output directory: {OUTPUT_SAM_DIR}")
    shutil.rmtree(OUTPUT_SAM_DIR)
os.makedirs(OUTPUT_SAM_DIR, exist_ok=True)

# ==========================================================
# INITIALIZE SAM
# ==========================================================
print("🚀 Loading SAM model...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print(f"✅ SAM loaded on device: {DEVICE}\n")

# ==========================================================
# HELPER FUNCTION
# ==========================================================
def parse_voc_boxes(xml_path):
    """Parse Pascal VOC XML annotations and return boxes and labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bnd = obj.find("bndbox")
        x1 = int(float(bnd.find("xmin").text))
        y1 = int(float(bnd.find("ymin").text))
        x2 = int(float(bnd.find("xmax").text))
        y2 = int(float(bnd.find("ymax").text))
        boxes.append((x1, y1, x2, y2))
        labels.append(label)
    filename = root.find("filename").text
    return filename, boxes, labels

# ==========================================================
# MAIN LOOP
# ==========================================================
xml_files = [f for f in os.listdir(VOC_DIR) if f.endswith(".xml")]
if not xml_files:
    print(f"⚠️ No annotation files found in {VOC_DIR}")
    exit()

print(f"Found {len(xml_files)} annotation files in {VOC_DIR}\n")
for xml_file in tqdm(xml_files, desc="Generating SAM masks"):
    xml_path = os.path.join(VOC_DIR, xml_file)
    filename, boxes, labels = parse_voc_boxes(xml_path)

    img_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {filename}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Unable to read image: {filename}")
        continue

    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_h, img_w = image.shape[:2]
    merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Process each bounding box
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, logits = predictor.predict(
            box=input_box,
            multimask_output=False
        )

        mask = masks[0].astype(np.uint8)

        # Keep only the largest connected component
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (label_map == largest_label).astype(np.uint8)

        # Merge all masks
        merged_mask = cv2.bitwise_or(merged_mask, mask)

        # Optionally visualize
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save mask and annotated image
    base_name = os.path.splitext(filename)[0]
    mask_path = os.path.join(OUTPUT_SAM_DIR, f"{base_name}_mask.png")
    annotated_path = os.path.join(OUTPUT_SAM_DIR, f"{base_name}_annotated.png")

    cv2.imwrite(mask_path, merged_mask * 255)
    # cv2.imwrite(annotated_path, image)

# ==========================================================
# DONE
# ==========================================================
print("\n✅ Done!")
print(f"All SAM masks and annotated images saved in: {OUTPUT_SAM_DIR}")
