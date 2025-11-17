import os
import shutil
import torch
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from groundingdino.util.inference import Model as GroundingDINOModel

# ==========================================================
# CONFIGURATION
# ==========================================================
HOME = ""
DATA_DIR = os.path.join(HOME, "data")
ANNOTATIONS_DIR = os.path.join(HOME, "output-dino/annotations")
VOC_DIR = os.path.join(HOME, "output-dino/voc_annotations")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "only text on lower right corner"

print("\n========================================")
print("GroundingDINO – Text-Prompt Based Detection")
print("========================================")
print(f"Device: {DEVICE}")
print(f"Prompt: {TEXT_PROMPT}")
print(f"Data directory: {DATA_DIR}")
print("========================================\n")

# ==========================================================
# CLEAN OLD DIRECTORIES
# ==========================================================
for base_dir in [ANNOTATIONS_DIR, VOC_DIR]:
    if os.path.exists(base_dir):
        print(f"🧹 Removing old directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

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
# HELPER – EXTRACT REAL BOX COORDINATES
# ==========================================================
def extract_box_coordinates(box):
    """
    Safely extracts [x1, y1, x2, y2] from any structure that GroundingDINO returns.
    Supports tensor, list, numpy, nested list, etc.
    """
    try:
        # Tensor -> numpy
        if torch.is_tensor(box):
            box = box.detach().cpu().numpy()

        # Flatten deeply nested lists
        if isinstance(box, (list, tuple)):
            flat = []
            for item in box:
                if isinstance(item, (list, tuple, np.ndarray, torch.Tensor)):
                    arr = np.array(item).flatten().tolist()
                    flat.extend(arr)
                elif isinstance(item, (float, int)):
                    flat.append(float(item))
            box = flat

        box = np.array(box).astype(float).flatten()

        # Find 4 plausible numeric values
        valid = [v for v in box if np.isfinite(v)]
        if len(valid) < 4:
            return None
        return valid[:4]
    except Exception as e:
        return None

# ==========================================================
# MAIN LOOP
# ==========================================================
print("Processing images with text prompt...\n")

for img_path in tqdm(image_paths, desc="Detecting"):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)

    # Run prediction
    result = grounding_dino_model.predict_with_caption(
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Handle possible return formats
    if isinstance(result, tuple):
        if len(result) == 3:
            boxes, logits, phrases = result
        elif len(result) == 2:
            boxes, phrases = result
            logits = [1.0] * len(boxes)
        else:
            print(f"⚠️ Unexpected format ({len(result)}) for {filename}, skipping.")
            continue
    else:
        print(f"⚠️ Invalid return type ({type(result)}) for {filename}, skipping.")
        continue

    if boxes is None or len(boxes) == 0:
        print(f"⚠️ No detections for {filename}")
        continue

    out_boxes, out_labels = [], []
    for idx, (box, phrase) in enumerate(zip(boxes, phrases)):
        coords = extract_box_coordinates(box)
        if coords is None or len(coords) < 4:
            print(f"⚠️ Box {idx} malformed in {filename} → skipping.")
            continue

        x1, y1, x2, y2 = map(int, coords)
        if x2 <= x1 or y2 <= y1:
            continue

        out_boxes.append((x1, y1, x2, y2))
        out_labels.append("text")

        # Visualization
        label_text = phrase.strip() if isinstance(phrase, str) else "text"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label_text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not out_boxes:
        print(f"⚠️ No valid boxes kept for {filename}")
        continue

    # Save outputs
    annotated_path = os.path.join(ANNOTATIONS_DIR, f"{os.path.splitext(filename)[0]}.jpg")
    voc_path = os.path.join(VOC_DIR, f"{os.path.splitext(filename)[0]}.xml")

    cv2.imwrite(annotated_path, image)
    save_voc_annotation(img_path, out_boxes, out_labels, voc_path)

# ==========================================================
# DONE
# ==========================================================
print("\n✅ Done!")
print(f"Annotated images → {ANNOTATIONS_DIR}")
print(f"Pascal VOC annotations → {VOC_DIR}")
