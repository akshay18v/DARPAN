import os
import cv2
import pytesseract

# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "yolo_dataset",
    "images",
    "misdirection_2.3.png"
)

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "yolo_dataset",
    "labels",
    "misdirection_2.3.txt"
)

# ==========================================
# TESSERACT
# ==========================================

pytesseract.pytesseract.tesseract_cmd = (
    r"D:\Tesseract-OCR\tesseract.exe"
)

# ==========================================
# READ IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image not found!")
    print(IMAGE_PATH)
    exit()

height, width = image.shape[:2]

print("=" * 60)
print("OCR CROP TEST")
print("=" * 60)

print("Image:", IMAGE_PATH)
print("Image size:", width, "x", height)

# ==========================================
# READ ALL YOLO LABELS
# ==========================================

with open(LABEL_PATH, "r") as f:
    lines = f.readlines()

print("Number of annotated regions:", len(lines))

# ==========================================
# PROCESS EACH ANNOTATED REGION
# ==========================================

for i, line in enumerate(lines):

    parts = line.strip().split()

    if len(parts) != 5:
        continue

    class_id = int(parts[0])

    x_center = float(parts[1])
    y_center = float(parts[2])
    box_width = float(parts[3])
    box_height = float(parts[4])

    # ======================================
    # YOLO → PIXEL COORDINATES
    # ======================================

    x_center = x_center * width
    y_center = y_center * height

    box_width = box_width * width
    box_height = box_height * height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)

    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Keep inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    print("\n" + "-" * 60)
    print("REGION:", i + 1)
    print("Class ID:", class_id)
    print("Coordinates:", x1, y1, x2, y2)

    # ======================================
    # CROP
    # ======================================

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print("❌ Empty crop")
        continue

    # ======================================
    # OCR PREPROCESSING
    # ======================================

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Enlarge
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # ======================================
    # OCR
    # ======================================

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    text = text.strip()

    # ======================================
    # OUTPUT
    # ======================================

    print("EXTRACTED TEXT:")

    if text:
        print(text)
    else:
        print("❌ No text detected")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)