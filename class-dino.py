import os
import shutil
import torch
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import supervision as sv
from groundingdino.util.inference import Model as GroundingDINOModel

# ==========================================================
# CONFIGURATION
# ==========================================================
HOME = ""
DATA_DIR = os.path.join(HOME, "data")
ANNOTATIONS_DIR = os.path.join(HOME, "output-dino/annotations")
VOC_DIR = os.path.join(HOME, "output-dino/voc_annotations")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15
CLASSES = ["text"]

print("\n========================================")
print("GroundingDINO – Bounding Boxes + Pascal VOC Export")
print("========================================")
print(f"Device: {DEVICE}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"ANNOTATIONS_DIR: {ANNOTATIONS_DIR}")
print(f"VOC_DIR: {VOC_DIR}")
print("========================================\n")

# ==========================================================
# CLEAN OLD DIRECTORIES
# ==========================================================
for base_dir in [ANNOTATIONS_DIR, VOC_DIR]:
    if os.path.exists(base_dir):
        print(f"🧹 Removing old directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

# ==========================================================
# INITIALIZE MODEL
# ==========================================================
print("Loading GroundingDINO...")
grounding_dino_model = GroundingDINOModel(
    model_config_path=os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    model_checkpoint_path=os.path.join(HOME, "GroundingDINO/weights/groundingdino_swint_ogc.pth"),
    device=DEVICE
)

# ==========================================================
# DISCOVER IMAGES
# ==========================================================
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    image_paths.extend(glob.glob(os.path.join(DATA_DIR, ext)))

if not image_paths:
    print("⚠️ No images found.")
    exit()

print(f"Found {len(image_paths)} image(s) in {DATA_DIR}\n")

# ==========================================================
# HELPER – SAVE PASCAL VOC XML
# ==========================================================
def save_voc_annotation(image_path, boxes, labels, save_path):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    filename = os.path.basename(image_path)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    for (box, label) in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    tree = ET.ElementTree(annotation)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tree.write(save_path)

# ==========================================================
# MAIN PROCESS
# ==========================================================
print("Processing images...\n")

for img_path in tqdm(image_paths, desc="Processing"):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    if len(detections) == 0:
        print(f"⚠️ No detections for {filename}")
        continue

    boxes, labels = [], []
    for box, label_id in zip(detections.xyxy, detections.class_id):
        cls_name = CLASSES[int(label_id)] if isinstance(label_id, (int, np.integer)) else str(label_id)
        x1, y1, x2, y2 = map(int, box)
        boxes.append((x1, y1, x2, y2))
        labels.append(cls_name)

        # draw bbox on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, cls_name, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # save annotated image (per class)
    for cls_name in CLASSES:
        cls_dir = os.path.join(ANNOTATIONS_DIR, cls_name)
        cv2.imwrite(os.path.join(cls_dir, filename), image)

    # save Pascal VOC
    voc_path = os.path.join(VOC_DIR, cls_name, f"{os.path.splitext(filename)[0]}.xml")
    save_voc_annotation(img_path, boxes, labels, voc_path)

# ==========================================================
# DONE
# ==========================================================
print("\n✅ Done!")
print(f"Annotated images → {ANNOTATIONS_DIR}")
print(f"Pascal VOC annotations → {VOC_DIR}")
