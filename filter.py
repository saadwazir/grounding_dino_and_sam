import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_DIR = ""
DATA_DIR = os.path.join(BASE_DIR, "data")
VOC_DIR = os.path.join(BASE_DIR, "output-dino", "voc_annotations")

FILTER_DIR = os.path.join(BASE_DIR, "output-filter")
OUTPUT_VOC_DIR = os.path.join(FILTER_DIR, "filtered_voc")
OUTPUT_IMG_DIR = os.path.join(FILTER_DIR, "filtered_images")

# Filtering thresholds
MAX_REL_AREA = 0.01        # reject if covers >1% of full image
MIN_WIDTH = 20             # px
MIN_HEIGHT = 10            # px
MIN_ASPECT_RATIO = 1.2     # w/h ratio (text usually wider)
SQUARE_TOLERANCE = 0.15    # ±15% margin around square ratio (1.0)
SAVE_EMPTY = True          # keep XML even if all boxes removed

# ==========================================================
# CLEAN / CREATE OUTPUT STRUCTURE
# ==========================================================
if os.path.exists(FILTER_DIR):
    print(f"🧹 Removing old filter directory: {FILTER_DIR}")
    shutil.rmtree(FILTER_DIR)

os.makedirs(OUTPUT_VOC_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def parse_voc(xml_path):
    """Parse Pascal VOC XML and return boxes, labels, and XML tree."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes, labels = [], []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bnd = obj.find("bndbox")
        x1 = int(float(bnd.find("xmin").text))
        y1 = int(float(bnd.find("ymin").text))
        x2 = int(float(bnd.find("xmax").text))
        y2 = int(float(bnd.find("ymax").text))
        boxes.append((x1, y1, x2, y2))
        labels.append(label)

    return filename, boxes, labels, tree, root


def filter_boxes(image, boxes, labels):
    """Apply geometric and contextual filtering to bounding boxes."""
    h, w = image.shape[:2]
    kept_boxes, kept_labels = [], []

    # Estimate visible (non-black) area to normalize box area correctly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzero_mask = (gray > 10).astype("uint8")
    fg_area = cv2.countNonZero(nonzero_mask)
    fg_ratio = fg_area / (w * h + 1e-5)

    for (x1, y1, x2, y2), lbl in zip(boxes, labels):
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        rel_area_img = area / (w * h + 1e-5)
        rel_area_fg = area / (fg_area + 1e-5)
        aspect_ratio = bw / (bh + 1e-5)

        # Square detection (1.0 ± tolerance)
        is_square = abs(aspect_ratio - 1.0) <= SQUARE_TOLERANCE

        # ----------- Filtering Logic -----------
        too_large_global = rel_area_img > MAX_REL_AREA
        too_large_foreground = rel_area_fg > 0.5       # covers >50% of visible area
        absurdly_big = area > (0.5 * w * h)            # absolute cutoff
        too_small = (bw < MIN_WIDTH) or (bh < MIN_HEIGHT)
        too_tall = (aspect_ratio < MIN_ASPECT_RATIO and bh > bw)
        too_square = is_square                         # new condition

        if (
            too_large_global
            or too_large_foreground
            or absurdly_big
            or too_small
            or too_tall
            or too_square
        ):
            print(
                f"🟥 Removed {lbl} box ({bw}x{bh}) | AR:{aspect_ratio:.2f} | rel_area:{rel_area_img:.3f} | rel_fg:{rel_area_fg:.3f}"
            )
            continue

        kept_boxes.append((x1, y1, x2, y2))
        kept_labels.append(lbl)

    return kept_boxes, kept_labels


def update_voc(tree, root, boxes, labels, save_path):
    """Write filtered Pascal VOC XML file."""
    for obj in list(root.findall("object")):
        root.remove(obj)

    for (x1, y1, x2, y2), lbl in zip(boxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = lbl
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tree.write(save_path)

# ==========================================================
# MAIN LOOP
# ==========================================================
print(f"🔍 Filtering Pascal VOC annotations in: {VOC_DIR}\n")

xml_files = [f for f in os.listdir(VOC_DIR) if f.endswith(".xml")]
if not xml_files:
    print("⚠️ No XML files found.")
    exit()

for xml_file in tqdm(xml_files, desc="Processing"):
    xml_path = os.path.join(VOC_DIR, xml_file)
    filename, boxes, labels, tree, root = parse_voc(xml_path)

    img_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image for {xml_file}")
        continue

    image = cv2.imread(img_path)
    kept_boxes, kept_labels = filter_boxes(image, boxes, labels)

    filtered_img = image.copy()
    for (x1, y1, x2, y2), lbl in zip(kept_boxes, kept_labels):
        cv2.rectangle(filtered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(filtered_img, lbl, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if kept_boxes or SAVE_EMPTY:
        update_voc(
            tree, root, kept_boxes, kept_labels,
            os.path.join(OUTPUT_VOC_DIR, xml_file)
        )
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename), filtered_img)
    else:
        print(f"🗑️ Skipped {filename} (no valid boxes kept)")

print("\n✅ Done!")
print(f"Filtered XMLs → {OUTPUT_VOC_DIR}")
print(f"Filtered visualization images → {OUTPUT_IMG_DIR}")
